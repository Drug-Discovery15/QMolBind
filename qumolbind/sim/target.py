"""Locate and load a prepared target (data_cache/targets/<id> or the committed 1cil test fixture)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from qumolbind.chem.ligand import LigandModel
from qumolbind.chem.rmsd import SymmetryRMSD
from qumolbind.sim.oracle_fast import FastOracle

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Target:
    target_id: str
    protein_pdb: Path
    ligand: LigandModel
    rmsd: SymmetryRMSD

    def make_oracle(self, **kw) -> FastOracle:
        return FastOracle(self.protein_pdb, self.ligand.mol, self.ligand.native, **kw)


def find_target_files(target_id: str) -> tuple[Path, Path]:
    d = ROOT / "data_cache" / "targets" / target_id
    if (d / "protein.pdb").exists():
        return d / "protein.pdb", d / "ligand_native.sdf"
    fx = ROOT / "tests" / "fixtures"
    if (fx / f"{target_id}_protein.pdb").exists():
        return fx / f"{target_id}_protein.pdb", fx / f"{target_id}_ligand_native.sdf"
    raise FileNotFoundError(f"target {target_id!r} not prepared; run scripts/prep_targets.py")


def load_target(target_id: str, max_torsions: int = 8) -> Target:
    prot, sdf = find_target_files(target_id)
    mol = Chem.MolFromMolFile(str(sdf), removeHs=False)
    lm = LigandModel(mol, max_torsions)
    return Target(target_id, prot, lm, SymmetryRMSD(lm.mol, lm.native))
