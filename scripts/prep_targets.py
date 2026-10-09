"""Stage 2: prepare every screened-and-accepted target and print torsions + native-pose sanity checks."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.chem.ligand import LigandModel  # noqa: E402
from qumolbind.chem.prep import TARGET_DIR, prepare_target  # noqa: E402
from qumolbind.chem.rmsd import SymmetryRMSD  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["1cil", "3ert", "1uyd"])
    ap.add_argument("--radius", type=float, default=12.0)
    ap.add_argument("--max-torsions", type=int, default=8)
    args = ap.parse_args()
    screen = {c["pdb_id"].lower(): c for c in json.loads((ROOT / "data_cache" / "screen.json").read_text())}
    for tid in args.targets:
        c = screen[tid]
        t = prepare_target(c["pdb_id"], c["ligand_id"], c["ligand_chain"], c["ligand_resnum"], c["smiles"], pocket_radius=args.radius)
        mol = Chem.MolFromMolFile(str(t.ligand_sdf), removeHs=False)
        lm = LigandModel(mol, args.max_torsions)
        rm = SymmetryRMSD(mol, lm.native)
        rng = np.random.default_rng(0)
        rand = lm.randomize(rng)
        back = lm.set_torsions_deg(rand, lm.get_torsions_deg(lm.native))
        print(f"=== {tid} {c['ligand_id']}  protein: {t.meta['n_protein_residues']} res / {t.meta['n_protein_atoms']} atoms (R={args.radius} A)")
        print(lm.describe())
        print(f"  native RMSD(native)={rm(lm.native):.6f}  random-torsion RMSD={rm(rand):.2f}  restore-to-native RMSD={rm(back):.2e}")
        # heavy atom min distance to protein in native pose (sanity: no clash, in contact)
        prot = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in (t.protein_pdb).read_text().splitlines() if l.startswith(("ATOM", "HETATM"))])
        d = np.linalg.norm(prot[None] - lm.native[lm.heavy_idx][:, None], axis=-1)
        print(f"  native min heavy-atom/protein distance = {d.min():.2f} A; contacts<4A: {(d.min(0) < 4).sum()} protein atoms")
        if tid == "1cil":
            shutil.copy(t.protein_pdb, FIXTURES / "1cil_protein.pdb")
            shutil.copy(t.ligand_sdf, FIXTURES / "1cil_ligand_native.sdf")


if __name__ == "__main__":
    main()
