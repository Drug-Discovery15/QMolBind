"""Level-1 fast oracle: ligand-protein interaction energy in implicit solvent, protein frozen.

E_int = E_complex - E_protein - E_ligand  (all with amber14 protein + generated ligand FF + GBn2)
      = E_vdw + E_elec(vacuum Coulomb) + E_solv(delta GBn2)

vdW and Coulomb between ligand and protein atoms are evaluated analytically in numpy with the exact
parameters OpenMM holds (Lorentz-Berthelot mixing, no cutoff); the solvation term is the GBn2
(CustomGBForce) energy difference computed by OpenMM. ``reference_interaction`` recomputes E_int from
full OpenMM energies of the three systems and is used in tests to prove the decomposition.
Lower is better. ``score`` additionally adds MMFF94 intramolecular strain (self-clashes / torsion
strain), which the pure interaction energy cannot see (D12 in DECISIONS.md).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit
from rdkit import Chem

from qumolbind.sim.params import (
    LigandParams, StrainEnergy, ligand_forcefield_xml, ligand_topology, parametrize_ligand,
)

ONE_4PI_EPS0 = 138.93545764438198  # kJ nm / (mol e^2)
GB_GROUP, NB_GROUP = 2, 1


@dataclass
class EnergyTerms:
    vdw: float
    elec: float
    solv: float
    strain: float
    e_int: float     # vdw + elec + solv
    score: float     # e_int + strain (what search/RL minimises)

    def terms(self) -> np.ndarray:
        return np.array([self.vdw, self.elec, self.solv])


def select_platform(name: str = "auto", precision: str = "mixed", threads: int = 1) -> tuple[openmm.Platform, dict[str, str]]:
    """OpenCL (GPU) is ~1000x faster than the single-thread CPU platform for GBn2 here (see docs/ARCHITECTURE.md)."""
    names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    if name == "auto":
        name = "OpenCL" if "OpenCL" in names else "CPU"
    if name == "OpenCL":
        return openmm.Platform.getPlatformByName("OpenCL"), {"Precision": precision}
    if name == "CPU":
        return openmm.Platform.getPlatformByName("CPU"), {"Threads": str(threads)}
    return openmm.Platform.getPlatformByName(name), {}


class FastOracle:
    def __init__(
        self,
        protein_pdb: str | Path,
        ligand_mol: Chem.Mol,
        native_coords: np.ndarray,
        include_strain: bool = True,
        minimize_iters: int = 0,
        threads: int = 1,
        platform: str = "auto",
        precision: str = "mixed",
        ff_files: tuple[str, ...] = ("amber14-all.xml", "amber14/tip3p.xml", "implicit/gbn2.xml"),
    ) -> None:
        self.mol = Chem.Mol(ligand_mol)
        self.n_lig = self.mol.GetNumAtoms()
        self.include_strain = include_strain
        self.minimize_iters = minimize_iters
        self.calls = 0
        self.params: LigandParams = parametrize_ligand(self.mol)
        self.strain = StrainEnergy(self.mol, native_coords)

        pdb = app.PDBFile(str(protein_pdb))
        self.protein_positions_nm = np.array(pdb.positions.value_in_unit(unit.nanometer))
        self.n_prot = pdb.topology.getNumAtoms()
        lig_top = ligand_topology(self.mol)
        xml = ligand_forcefield_xml(self.mol, self.params, native_coords)
        ff = app.ForceField(*ff_files, io.StringIO(xml))
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.add(lig_top, openmm.unit.Quantity(np.asarray(native_coords) / 10.0, unit.nanometer))
        kw = dict(nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False)
        self.sys_c = ff.createSystem(mod.topology, **kw)
        self.sys_p = ff.createSystem(pdb.topology, **kw)
        self.sys_l = ff.createSystem(lig_top, **kw)
        plat, props = select_platform(platform, precision, threads)
        self.platform_name = plat.getName()
        self.ctx_c, self.ctx_p, self.ctx_l = (self._context(s, plat, props) for s in (self.sys_c, self.sys_p, self.sys_l))
        # pair parameters for the analytic ligand-protein vdW / Coulomb
        nb = next(f for f in self.sys_c.getForces() if isinstance(f, openmm.NonbondedForce))
        prm = np.array([[*[v.value_in_unit_system(unit.md_unit_system) for v in nb.getParticleParameters(i)]] for i in range(self.n_prot + self.n_lig)])
        self.q_p, self.s_p, self.e_p = prm[: self.n_prot, 0], prm[: self.n_prot, 1], prm[: self.n_prot, 2]
        self.q_l, self.s_l, self.e_l = prm[self.n_prot :, 0], prm[self.n_prot :, 1], prm[self.n_prot :, 2]
        self.sig_pair = 0.5 * (self.s_l[:, None] + self.s_p[None])
        self.eps_pair = np.sqrt(self.e_l[:, None] * self.e_p[None])
        self.qq_pair = ONE_4PI_EPS0 * self.q_l[:, None] * self.q_p[None]
        # frozen protein reference energy (constant)
        self.ctx_p.setPositions(self.protein_positions_nm)
        self.gb_p = self._energy(self.ctx_p, {GB_GROUP})
        for i in range(self.n_prot):  # zero mass => frozen under minimisation
            self.sys_c.setParticleMass(i, 0.0)
        self.ctx_c.reinitialize()
        self.native_coords = np.array(native_coords, dtype=np.float64)

    @staticmethod
    def _context(system: openmm.System, plat: openmm.Platform, props: dict) -> openmm.Context:
        for f in system.getForces():
            f.setForceGroup(NB_GROUP if isinstance(f, openmm.NonbondedForce) else GB_GROUP if isinstance(f, openmm.CustomGBForce) else 0)
        return openmm.Context(system, openmm.VerletIntegrator(0.001), plat, props)

    @staticmethod
    def _energy(ctx: openmm.Context, groups: set[int] | None = None) -> float:
        st = ctx.getState(getEnergy=True, groups=groups if groups is not None else -1)
        return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # ------------------------------------------------------------------
    def _pair_terms(self, lig_nm: np.ndarray) -> tuple[float, float]:
        r = np.linalg.norm(lig_nm[:, None, :] - self.protein_positions_nm[None], axis=-1)
        sr6 = (self.sig_pair / r) ** 6
        vdw = float(np.sum(4.0 * self.eps_pair * (sr6 * sr6 - sr6)))
        coul = float(np.sum(self.qq_pair / r))
        return vdw, coul

    def evaluate(self, coords_angstrom: np.ndarray) -> EnergyTerms:
        """Score a ligand pose (N,3 Angstrom, protein frame). Counts one oracle call."""
        self.calls += 1
        lig_nm = np.asarray(coords_angstrom, dtype=np.float64) / 10.0
        if self.minimize_iters > 0:
            lig_nm = self._minimized(lig_nm)
        vdw, coul = self._pair_terms(lig_nm)
        self.ctx_c.setPositions(np.vstack([self.protein_positions_nm, lig_nm]))
        self.ctx_l.setPositions(lig_nm)
        solv = self._energy(self.ctx_c, {GB_GROUP}) - self.gb_p - self._energy(self.ctx_l, {GB_GROUP})
        strain = self.strain(lig_nm * 10.0) if self.include_strain else 0.0
        e_int = vdw + coul + solv
        return EnergyTerms(vdw, coul, solv, strain, e_int, e_int + strain)

    def _minimized(self, lig_nm: np.ndarray) -> np.ndarray:
        self.ctx_c.setPositions(np.vstack([self.protein_positions_nm, lig_nm]))
        openmm.LocalEnergyMinimizer.minimize(self.ctx_c, 10.0, self.minimize_iters)
        pos = self.ctx_c.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return np.array(pos[self.n_prot :])

    def reference_interaction(self, coords_angstrom: np.ndarray) -> float:
        """E_complex - E_protein - E_ligand from FULL OpenMM energies (all force groups). For tests."""
        lig_nm = np.asarray(coords_angstrom, dtype=np.float64) / 10.0
        self.ctx_c.setPositions(np.vstack([self.protein_positions_nm, lig_nm]))
        self.ctx_l.setPositions(lig_nm)
        self.ctx_p.setPositions(self.protein_positions_nm)
        return self._energy(self.ctx_c) - self._energy(self.ctx_p) - self._energy(self.ctx_l)

    def benchmark(self, n: int = 50, rng_seed: int = 0) -> float:
        """Evaluations per second on native-neighbourhood poses (single thread)."""
        rng = np.random.default_rng(rng_seed)
        poses = [self.native_coords + rng.normal(0, 0.05, self.native_coords.shape) for _ in range(n)]
        t0 = time.perf_counter()
        for p in poses:
            self.evaluate(p)
        return n / (time.perf_counter() - t0)
