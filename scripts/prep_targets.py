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
    ap.add_argument("--targets", nargs="*", default=["3ert", "1uyd", "1eve", "1cil"])
    ap.add_argument("--radius", type=float, default=12.0)
    ap.add_argument("--max-torsions", type=int, default=8)
    args = ap.parse_args()
    screen = {c["pdb_id"].lower(): c for c in json.loads((ROOT / "data_cache" / "screen.json").read_text())}
    param_rows = []
    for tid in args.targets:
        c = screen[tid]
        t = prepare_target(c["pdb_id"], c["ligand_id"], c["ligand_chain"], c["ligand_resnum"], c["smiles"], pocket_radius=args.radius)
        mol = Chem.MolFromMolFile(str(t.ligand_sdf), removeHs=False)
        lm = LigandModel(mol, args.max_torsions)
        rm = SymmetryRMSD(mol, lm.native)
        rng = np.random.default_rng(0)
        rand = lm.randomize(rng)
        back = lm.set_torsions_deg(rand, lm.get_torsions_deg(lm.native))
        try:  # ligand parametrisation: catch, log, report (pitfall: odd atom types / charge failures)
            from qumolbind.sim.params import parametrize_ligand

            pr = parametrize_ligand(lm.mol)
            param_rows.append({"target": tid, "ligand": c["ligand_id"], "ok": True, "charge_method": pr.charge_method, "net_charge": round(float(pr.charges.sum()), 6), "error": ""})
        except Exception as e:  # noqa: BLE001
            param_rows.append({"target": tid, "ligand": c["ligand_id"], "ok": False, "charge_method": "", "net_charge": float("nan"), "error": str(e)[:200]})
        print(f"=== {tid} {c['ligand_id']}  protein: {t.meta['n_protein_residues']} res / {t.meta['n_protein_atoms']} atoms (R={args.radius} A)")
        print(lm.describe())
        print(f"  native RMSD(native)={rm(lm.native):.6f}  random-torsion RMSD={rm(rand):.2f}  restore-to-native RMSD={rm(back):.2e}")
        # heavy atom min distance to protein in native pose (sanity: no clash, in contact)
        prot = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in (t.protein_pdb).read_text().splitlines() if l.startswith(("ATOM", "HETATM"))])
        d = np.linalg.norm(prot[None] - lm.native[lm.heavy_idx][:, None], axis=-1)
        print(f"  native min heavy-atom/protein distance = {d.min():.2f} A; contacts<4A: {(d.min(0) < 4).sum()} protein atoms")
        if tid in ("1cil", "3ert"):
            shutil.copy(t.protein_pdb, FIXTURES / f"{tid}_protein.pdb")
            shutil.copy(t.ligand_sdf, FIXTURES / f"{tid}_ligand_native.sdf")
    write_param_report(param_rows)


def write_param_report(rows: list[dict]) -> None:
    import pandas as pd

    if rows:
        df = pd.DataFrame(rows)
        (ROOT / "results").mkdir(exist_ok=True)
        df.to_csv(ROOT / "results" / "parametrization_report.csv", index=False)
        print(f"ligand parametrisation: {int(df.ok.sum())}/{len(df)} succeeded ({100 * (1 - df.ok.mean()):.0f}% failure rate)")


if __name__ == "__main__":
    main()
