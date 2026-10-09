"""RCSB download + programmatic screening of co-crystal complexes."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

RCSB_FILES = "https://files.rcsb.org/download/{pid}.pdb"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pid}"
RCSB_CHEMCOMP = "https://data.rcsb.org/rest/v1/core/chemcomp/{cid}"

# Common crystallization additives / ions / buffers that are never "the" ligand.
EXCLUDED_HET = {
    "HOH", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "PG4", "MPD", "DMS", "ACT", "FMT", "TRS", "MES",
    "EPE", "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "CD", "NI", "CO", "CU", "IOD", "BR", "NO3", "SCN",
    "ACE", "NH2", "BME", "CIT", "TAR", "IPA", "EOH", "MOH", "PEG", "1PE", "P6G", "PE4", "BOG", "OCT", "NAG",
}
RESULTS_DIR = Path(__file__).resolve().parents[2] / "data_cache"


@dataclass
class Candidate:
    pdb_id: str
    resolution: float | None = None
    ligand_id: str | None = None
    ligand_chain: str | None = None
    ligand_resnum: int | None = None
    smiles: str | None = None
    n_heavy: int | None = None
    mol_wt: float | None = None
    n_rotors: int | None = None
    covalent: bool | None = None
    n_candidate_ligands: int = 0
    uniprot: list[str] = field(default_factory=list)
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)


def _get(url: str, timeout: float = 60) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def download_pdb(pdb_id: str, out_dir: str | Path = RESULTS_DIR / "pdb") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{pdb_id.upper()}.pdb"
    if not path.exists():
        path.write_text(_get(RCSB_FILES.format(pid=pdb_id.upper())).text)
    return path


def fetch_chemcomp_smiles(ligand_id: str) -> str:
    d = _get(RCSB_CHEMCOMP.format(cid=ligand_id)).json()
    desc = d.get("rcsb_chem_comp_descriptor", {})
    smi = desc.get("SMILES_stereo") or desc.get("SMILES")
    if not smi:
        raise ValueError(f"no SMILES for CCD {ligand_id}")
    return smi


def parse_hetero_groups(pdb_text: str) -> dict[tuple[str, str, int], list[tuple[str, np.ndarray]]]:
    """Group HETATM records by (resname, chain, resnum) -> [(atom_name, xyz)]."""
    groups: dict[tuple[str, str, int], list[tuple[str, np.ndarray]]] = {}
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            res = line[17:20].strip()
            if res in EXCLUDED_HET:
                continue
            key = (res, line[21], int(line[22:26]))
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            groups.setdefault(key, []).append((line[12:16].strip(), xyz))
    return groups


def _protein_coords(pdb_text: str) -> np.ndarray:
    xyz = [
        [float(l[30:38]), float(l[38:46]), float(l[46:54])]
        for l in pdb_text.splitlines()
        if l.startswith("ATOM")
    ]
    return np.array(xyz)


def screen_candidate(
    pdb_id: str,
    max_resolution: float = 2.5,
    rotors: tuple[int, int] = (3, 8),
    heavy: tuple[int, int] = (10, 45),
) -> Candidate:
    """Programmatically verify a candidate against the selection filter."""
    pid = pdb_id.upper()
    c = Candidate(pdb_id=pid)
    entry = _get(RCSB_ENTRY.format(pid=pid)).json()
    res = entry.get("rcsb_entry_info", {}).get("resolution_combined")
    c.resolution = float(res[0]) if res else None
    if c.resolution is None or c.resolution > max_resolution:
        c.reasons.append(f"resolution {c.resolution} > {max_resolution}")
    text = download_pdb(pid).read_text()
    groups = parse_hetero_groups(text)
    # keep groups of non-trivial size, then collapse symmetric copies of the same CCD id
    big = {k: v for k, v in groups.items() if len(v) >= 5}
    ids = sorted({k[0] for k in big})
    c.n_candidate_ligands = len(ids)
    if len(ids) != 1:
        c.reasons.append(f"need exactly one ligand type, found {ids}")
        return c
    lig = ids[0]
    key = sorted(k for k in big if k[0] == lig)[0]
    c.ligand_id, c.ligand_chain, c.ligand_resnum = lig, key[1], key[2]
    try:
        c.smiles = fetch_chemcomp_smiles(lig)
    except Exception as e:  # network / missing
        c.reasons.append(f"no SMILES: {e}")
        return c
    mol = Chem.MolFromSmiles(c.smiles)
    if mol is None:
        c.reasons.append("SMILES unparsable")
        return c
    c.n_heavy = mol.GetNumHeavyAtoms()
    c.mol_wt = Descriptors.MolWt(mol)
    c.n_rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if not heavy[0] <= c.n_heavy <= heavy[1]:
        c.reasons.append(f"heavy atoms {c.n_heavy} outside {heavy}")
    if not rotors[0] <= c.n_rotors <= rotors[1]:
        c.reasons.append(f"rotors {c.n_rotors} outside {rotors}")
    # covalent: any LINK record involving the ligand and a polymer residue, or sub-2.0 A C/N/O/S-protein contact
    link = any(
        l.startswith("LINK") and lig in (l[17:20], l[47:50]) and not ({l[17:20].strip(), l[47:50].strip()} <= {lig})
        and not {l[12:16].strip(), l[42:46].strip()} & {"ZN", "MG", "CA", "NA", "K", "MN", "FE"}
        for l in text.splitlines()
    )
    prot = _protein_coords(text)
    heavy_xyz = np.array([x for n, x in big[key] if not n.startswith("H")])
    dmin = float(np.min(np.linalg.norm(prot[None] - heavy_xyz[:, None], axis=-1)))
    c.covalent = bool(link) or dmin < 1.7
    if c.covalent:
        c.reasons.append(f"covalent-like (LINK={link}, min contact {dmin:.2f} A)")
    for l in text.splitlines():
        if l.startswith("DBREF") and l[26:32].strip() == "UNP":
            c.uniprot.append(l[33:41].strip())
    c.uniprot = sorted(set(c.uniprot))
    c.accepted = not c.reasons
    return c


def screen_many(pdb_ids: list[str]) -> list[Candidate]:
    out = []
    for p in pdb_ids:
        try:
            out.append(screen_candidate(p))
        except Exception as e:
            out.append(Candidate(pdb_id=p.upper(), reasons=[f"error: {e}"]))
    return out


def save_screen(cands: list[Candidate], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(c) for c in cands], indent=2))
