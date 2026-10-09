"""Target preparation: protein clean-up (pdbfixer), pocket truncation with ACE/NME caps, ligand from crystal pose."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem

from qumolbind.data.pdb import download_pdb, fetch_chemcomp_smiles

KEEP_IONS = {"ZN", "MG", "CA", "MN"}  # structural/catalytic metals with amber14 templates (see DECISIONS.md)
TARGET_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "targets"


@dataclass
class PreparedTarget:
    target_id: str
    protein_pdb: Path
    ligand_sdf: Path
    meta: dict


def _hetatm_block(pdb_text: str, ligand_id: str, chain: str, resnum: int) -> str:
    lines = []
    for l in pdb_text.splitlines():
        if l.startswith("HETATM") and l[17:20] == ligand_id and l[21] == chain and int(l[22:26]) == resnum:
            if l[16] not in (" ", "A"):  # keep first altloc only
                continue
            lines.append(l[:16] + " " + l[17:])
    return "\n".join(lines) + "\nEND\n"


def ligand_from_crystal(pdb_text: str, ligand_id: str, chain: str, resnum: int, smiles: str | None = None) -> Chem.Mol:
    """Bond orders from the CCD template; hydrogens added; crystal heavy-atom coordinates kept (native pose)."""
    smiles = smiles or fetch_chemcomp_smiles(ligand_id)
    template = Chem.MolFromSmiles(smiles)
    pdb_mol = Chem.MolFromPDBBlock(_hetatm_block(pdb_text, ligand_id, chain, resnum), removeHs=True, sanitize=False)
    mol = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    mol.SetProp("_Name", ligand_id)
    return mol


def _fix_protein(pdb_path: Path, keep_chains: list[str]):
    fx = PDBFixer(filename=str(pdb_path))
    all_chains = [c.id for c in fx.topology.chains()]
    fx.removeChains(chainIds=[c for c in all_chains if c not in keep_chains])
    fx.findMissingResidues()
    fx.missingResidues = {}  # never invent loops; chain breaks are handled by truncation/caps
    fx.findNonstandardResidues()
    fx.replaceNonstandardResidues()
    fx.removeHeterogens(keepWater=False)
    fx.findMissingAtoms()
    fx.addMissingAtoms()
    return fx


def _ions_from(pdb_text: str, chains: list[str]) -> list[str]:
    return [l for l in pdb_text.splitlines() if l.startswith("HETATM") and l[17:20].strip() in KEEP_IONS and l[21] in chains]


def truncate_with_caps(pdb_lines: list[str], center_xyz: np.ndarray, radius: float) -> list[str]:
    """Keep residues with any heavy atom within ``radius`` of any ligand atom; cap broken ends with ACE/NME.

    ``pdb_lines`` are ATOM records of the fixed protein (heavy atoms + H allowed; H dropped here).
    Caps reuse the real coordinates of the removed neighbour residue (ACE <- CA,C,O of previous; NME <- N,CA of next),
    so geometry is native; hydrogens are re-added afterwards by pdbfixer.
    """
    res_atoms: dict[tuple[str, int], list[str]] = {}
    order: list[tuple[str, int]] = []
    for l in pdb_lines:
        if not l.startswith("ATOM") or l[76:78].strip() == "H" or l[12:16].strip().startswith("H"):
            continue
        k = (l[21], int(l[22:26]))
        if k not in res_atoms:
            res_atoms[k] = []
            order.append(k)
        res_atoms[k].append(l)
    xyz = lambda l: np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    near = {k for k, ls in res_atoms.items() if min(np.min(np.linalg.norm(center_xyz - xyz(l), axis=1)) for l in ls) <= radius}
    # fill short gaps (<= 3 residues) so caps never coincide and fragments stay contiguous
    sel = [k for k in order if k in near]
    for a, b in zip(sel, sel[1:]):
        ia, ib = order.index(a), order.index(b)
        if a[0] == b[0] and 1 < ib - ia <= 4 and b[1] - a[1] == ib - ia:
            near.update(order[ia + 1 : ib])
    out: list[str] = []
    serial = 1

    def emit(l: str, name: str | None = None, resname: str | None = None, resnum: int | None = None, chain: str | None = None) -> None:
        nonlocal serial
        nm = name if name is not None else l[12:16].strip()
        atom_field = f" {nm:<3s}" if len(nm) < 4 else nm
        out.append(
            f"{l[:6]}{serial:5d} {atom_field}{l[16]}{(resname or l[17:20]):>3s} {chain or l[21]}{(resnum if resnum is not None else int(l[22:26])):4d}{l[26:]}"
        )
        serial += 1

    idx = {k: i for i, k in enumerate(order)}
    for i, k in enumerate(order):
        if k not in near:
            continue
        prev_k = order[i - 1] if i > 0 and order[i - 1][0] == k[0] else None
        next_k = order[i + 1] if i + 1 < len(order) and order[i + 1][0] == k[0] else None
        consecutive_prev = prev_k is not None and prev_k[1] == k[1] - 1
        consecutive_next = next_k is not None and next_k[1] == k[1] + 1
        if consecutive_prev and prev_k not in near:  # N-terminal side of a segment -> ACE cap
            cap = {l[12:16].strip(): l for l in res_atoms[prev_k]}
            for src, dst in (("CA", "CH3"), ("C", "C"), ("O", "O")):
                emit(cap[src], name=dst, resname="ACE", resnum=k[1] - 1, chain=k[0])
        for l in res_atoms[k]:
            emit(l)
        if consecutive_next and next_k not in near:  # C-terminal side -> NME cap
            cap = {l[12:16].strip(): l for l in res_atoms[next_k]}
            for src, dst in (("N", "N"), ("CA", "CH3")):
                emit(cap[src], name=dst, resname="NME", resnum=k[1] + 1, chain=k[0])
        if k not in near or not (consecutive_next and next_k in near):
            out.append("TER")
    return out


def prepare_target(
    pdb_id: str,
    ligand_id: str,
    ligand_chain: str,
    ligand_resnum: int,
    smiles: str | None = None,
    pocket_radius: float | None = 12.0,
    ph: float = 7.4,
    out_root: Path = TARGET_DIR,
) -> PreparedTarget:
    """Produce ``protein.pdb`` (fixed, H added, optionally pocket-truncated+capped) and ``ligand_native.sdf``."""
    tid = pdb_id.lower()
    out = out_root / tid
    out.mkdir(parents=True, exist_ok=True)
    src = download_pdb(pdb_id)
    text = src.read_text()
    lig = ligand_from_crystal(text, ligand_id, ligand_chain, ligand_resnum, smiles)
    lig_xyz = lig.GetConformer().GetPositions()

    # chains with any heavy atom within 6 A of the ligand
    chains = set()
    for l in text.splitlines():
        if l.startswith("ATOM"):
            p = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            if np.min(np.linalg.norm(lig_xyz - p, axis=1)) <= 6.0:
                chains.add(l[21])
    keep = sorted(chains)
    fx = _fix_protein(src, keep)
    tmp = out / "_fixed_heavy.pdb"
    with open(tmp, "w") as f:
        PDBFile.writeFile(fx.topology, fx.positions, f, keepIds=True)
    lines = tmp.read_text().splitlines()
    ions = [l for l in _ions_from(text, keep)
            if np.min(np.linalg.norm(lig_xyz - np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), axis=1)) <= (pocket_radius or 1e9)]
    if pocket_radius is not None:
        body = truncate_with_caps(lines, lig_xyz, pocket_radius)
        trunc = out / "_trunc.pdb"
        trunc.write_text("\n".join(body + [l for l in ions] + ["END"]) + "\n")
        fx2 = PDBFixer(filename=str(trunc))
    else:
        full = out / "_full.pdb"
        full.write_text("\n".join([l for l in lines if l.startswith(("ATOM", "TER"))] + ions + ["END"]) + "\n")
        fx2 = PDBFixer(filename=str(full))
    fx2.findMissingResidues()
    fx2.missingResidues = {}
    fx2.findMissingAtoms()
    fx2.addMissingAtoms()
    fx2.addMissingHydrogens(ph)
    with open(out / "protein.pdb", "w") as f:
        PDBFile.writeFile(fx2.topology, fx2.positions, f, keepIds=True)
    for p in (tmp, out / "_trunc.pdb", out / "_full.pdb"):
        p.unlink(missing_ok=True)

    w = Chem.SDWriter(str(out / "ligand_native.sdf"))
    w.write(lig)
    w.close()
    meta = {
        "pdb_id": pdb_id, "ligand_id": ligand_id, "chains": keep, "pocket_radius": pocket_radius, "ph": ph,
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(lig)), "n_ligand_atoms": lig.GetNumAtoms(),
        "n_protein_atoms": fx2.topology.getNumAtoms(), "n_protein_residues": fx2.topology.getNumResidues(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return PreparedTarget(tid, out / "protein.pdb", out / "ligand_native.sdf", meta)
