"""Tiny offline fixtures under tests/fixtures (pocket-truncated real structures + a BindingDB slice)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qumolbind.data.pdb import download_pdb, parse_hetero_groups

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def pocket_pdb_text(pdb_text: str, ligand_id: str, radius: float = 12.0) -> str:
    """Keep ligand HETATMs and protein residues with any atom within ``radius`` A of the ligand."""
    groups = parse_hetero_groups(pdb_text)
    key = sorted(k for k in groups if k[0] == ligand_id)[0]
    lig_xyz = np.array([x for _, x in groups[key]])
    keep_res: set[tuple[str, str]] = set()
    atom_lines = []
    for l in pdb_text.splitlines():
        if l.startswith("ATOM"):
            xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            atom_lines.append((l, (l[21], l[22:27]), np.min(np.linalg.norm(lig_xyz - xyz, axis=1))))
    for _, rid, d in atom_lines:
        if d <= radius:
            keep_res.add(rid)
    out = [l for l, rid, _ in atom_lines if rid in keep_res]
    out += [
        l for l in pdb_text.splitlines()
        if l.startswith("HETATM") and l[17:20] == ligand_id and (l[21], int(l[22:26])) == (key[1], key[2])
    ]
    return "\n".join(out) + "\nEND\n"


def build_fixtures(pdb_id: str = "1CIL", ligand_id: str = "ETS", uniprot: str = "P00918", n_aff: int = 60) -> None:
    from qumolbind.data.bindingdb import fetch_raw

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    text = download_pdb(pdb_id).read_text()
    (FIXTURE_DIR / f"{pdb_id}_pocket.pdb").write_text(pocket_pdb_text(text, ligand_id))
    recs = fetch_raw(uniprot)[:n_aff]
    (FIXTURE_DIR / f"bindingdb_{uniprot}_tiny.json").write_text(json.dumps(recs))
