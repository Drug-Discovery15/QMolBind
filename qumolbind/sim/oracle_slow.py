"""Level-2 oracle: short implicit-solvent MD + single-trajectory MM-GBSA-style binding estimate.

    dG ~= < E_complex - E_protein - E_ligand >_frames

Terms INCLUDED (all from the same force field as Level 1, plus the heuristic ligand torsions that make MD meaningful):
  * ligand-protein vdW (LJ, UFF-derived ligand params, amber14 protein) and vacuum Coulomb (MMFF94 ligand charges)
  * GBn2 generalised-Born polar solvation difference AND the ACE non-polar (SASA) term (both live in the GB force)
  * ligand internal energy cancels exactly (same frame in complex and ligand-alone), as does the frozen protein energy
Terms NOT included: configurational / conformational entropy, explicit waters, protein relaxation (protein is frozen
for the whole study, mass 0), multiple trajectories. Averages use the second half of the trajectory (first half = equilibration).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import openmm
from openmm import app, unit
from rdkit import Chem

from qumolbind.sim.oracle_fast import select_platform
from qumolbind.sim.params import ligand_forcefield_xml, ligand_topology, parametrize_ligand


@dataclass
class SlowResult:
    dg: float            # mean over production frames, kJ/mol
    dg_sem: float        # standard error of the mean (naive, frames treated as independent)
    n_frames: int
    md_ps: float
    seconds: float
    final_coords: np.ndarray  # Angstrom, last frame
    minimized_dg: float       # first (post-minimisation) frame, for reference
    pre_relax_l1: float = float("nan")  # fast-oracle score of the pose handed to MD after torsion pre-relaxation


class SlowOracle:
    def __init__(self, protein_pdb, ligand_mol: Chem.Mol, native_coords: np.ndarray, fast_oracle=None, ligand_model=None,
                 platform: str = "auto",
                 precision: str = "mixed", temperature: float = 300.0, dt_ps: float = 0.002, report_ps: float = 0.5,
                 ff_files=("amber14-all.xml", "amber14/tip3p.xml", "implicit/gbn2.xml")) -> None:
        self.mol, self.T, self.dt, self.report_ps = Chem.Mol(ligand_mol), temperature, dt_ps, report_ps
        self.fast, self.lm = fast_oracle, ligand_model
        self.calls = 0
        params = parametrize_ligand(self.mol)
        xml = ligand_forcefield_xml(self.mol, params, native_coords, with_torsions=True)
        ff = app.ForceField(*ff_files, io.StringIO(xml))
        pdb = app.PDBFile(str(protein_pdb))
        self.n_prot, self.n_lig = pdb.topology.getNumAtoms(), self.mol.GetNumAtoms()
        self.prot_nm = np.array(pdb.positions.value_in_unit(unit.nanometer))
        lig_top = ligand_topology(self.mol)
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.add(lig_top, np.asarray(native_coords) / 10.0 * unit.nanometer)
        kw = dict(nonbondedMethod=app.NoCutoff, constraints=app.HBonds, rigidWater=False, removeCMMotion=False)
        self.sys_c = ff.createSystem(mod.topology, **kw)
        sys_p = ff.createSystem(pdb.topology, **kw)
        sys_l = ff.createSystem(lig_top, **kw)
        for i in range(self.n_prot):
            self.sys_c.setParticleMass(i, 0.0)  # frozen protein
        plat, props = select_platform(platform, precision)
        self.platform, self.props = plat, props
        self.ctx_p = openmm.Context(sys_p, openmm.VerletIntegrator(0.001), plat, props)
        self.ctx_l = openmm.Context(sys_l, openmm.VerletIntegrator(0.001), plat, props)
        self.ctx_p.setPositions(self.prot_nm)
        self.e_p = self.ctx_p.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def _dg_frame(self, ctx: openmm.Context, lig_nm: np.ndarray) -> float:
        e_c = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        self.ctx_l.setPositions(lig_nm)
        e_l = self.ctx_l.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return e_c - self.e_p - e_l

    def relax_torsions(self, coords: np.ndarray, n_evals: int = 150, seed: int = 0) -> tuple[np.ndarray, float]:
        """Greedy torsion-space descent on the Level-1 score (step 30 -> 4 deg). Needed because strained poses overflow the
        GPU fixed-point force accumulator (|F| saturates at 2^31 kJ/mol/nm) and cannot be minimised/integrated directly."""
        rng = np.random.default_rng(seed)
        x, fx = np.array(coords), self.fast.evaluate(coords).score
        for i in range(n_evals):
            step = 30.0 * (4.0 / 30.0) ** (i / max(n_evals - 1, 1))
            y = self.lm.apply_torsion_deltas(x, rng.normal(0, step, self.lm.K))
            fy = self.fast.evaluate(y).score
            if fy < fx:
                x, fx = y, fy
        return x, float(fx)

    def delta_g(self, coords_angstrom: np.ndarray, md_ps: float = 100.0, seed: int = 0, minimize_iters: int = 200,
                pre_relax_evals: int = 150) -> SlowResult:
        t0 = time.perf_counter()
        self.calls += 1
        pre = float("nan")
        if self.fast is not None and self.lm is not None and pre_relax_evals > 0:
            coords_angstrom, pre = self.relax_torsions(coords_angstrom, pre_relax_evals, seed)
            if pre > 1.5e3:  # still strained after relaxation -> MD would diverge; report as failed Level-2 evaluation
                return SlowResult(float("nan"), float("nan"), 0, 0.0, time.perf_counter() - t0, np.asarray(coords_angstrom), float("nan"), pre)
        integ = openmm.LangevinMiddleIntegrator(self.T * unit.kelvin, 1.0 / unit.picosecond, self.dt * unit.picoseconds)
        integ.setRandomNumberSeed(seed)
        ctx = openmm.Context(self.sys_c, integ, self.platform, self.props)
        ctx.setPositions(np.vstack([self.prot_nm, np.asarray(coords_angstrom) / 10.0]))
        for _ in range(8):  # staged minimisation: severe clashes need many iterations; stop once finite and relaxed
            openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, minimize_iters)
            e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if np.isfinite(e) and e - self.e_p < 1e5:
                break
        ctx.applyConstraints(1e-6)
        # gentle start: 0.5 fs steps while heating 30 K -> T over ~1 ps, then 2 fs production. Strained poses otherwise blow up.
        integ.setStepSize(0.0005 * unit.picoseconds)
        ctx.setVelocitiesToTemperature(30 * unit.kelvin, seed)
        for k in range(10):
            integ.setTemperature((30 + (self.T - 30) * (k + 1) / 10) * unit.kelvin)
            integ.step(200)
            if not np.isfinite(ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                return SlowResult(float("nan"), float("nan"), 0, 0.0, time.perf_counter() - t0, np.asarray(coords_angstrom), float("nan"), pre)
        integ.setStepSize(self.dt * unit.picoseconds)

        def lig_pos() -> np.ndarray:
            return np.array(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)[self.n_prot :])

        first = self._dg_frame(ctx, lig_pos())
        if not np.isfinite(first) or first > 1e5:  # could not be relaxed: report failure instead of integrating garbage
            return SlowResult(float("nan"), float("nan"), 0, 0.0, time.perf_counter() - t0, np.asarray(coords_angstrom), float(first), pre)
        steps_per_report = max(int(round(self.report_ps / self.dt)), 1)
        n_reports = max(int(round(md_ps / self.report_ps)), 2)
        frames = []
        for _ in range(n_reports):
            integ.step(steps_per_report)
            frames.append(self._dg_frame(ctx, lig_pos()))
        prod = np.array(frames[len(frames) // 2 :])
        final = lig_pos() * 10.0
        del ctx
        return SlowResult(
            dg=float(prod.mean()), dg_sem=float(prod.std(ddof=1) / np.sqrt(len(prod))) if len(prod) > 1 else float("nan"),
            n_frames=len(prod), md_ps=md_ps, seconds=time.perf_counter() - t0, final_coords=final, minimized_dg=float(first), pre_relax_l1=pre,
        )
