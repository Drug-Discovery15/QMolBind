"""Pocket embedding: ESM-2 (facebook/esm2_t6_8M_UR50D) over pocket residues, mean-pooled, fixed random projection to 32-d.

NOTE: for a single-target run this is a CONSTANT vector (same pocket every episode); it is kept for architecture
fidelity and `use_pocket_emb: false` is the default/ablation. The same holds for the ligand embedding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "facebook/esm2_t6_8M_UR50D"
OUT_DIM = 32
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H",
    "HID": "H", "HIP": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V", "CYX": "C", "ASH": "D", "GLH": "E",
}


def pocket_sequence(protein_pdb: str | Path) -> str:
    seen, seq = set(), []
    for l in Path(protein_pdb).read_text().splitlines():
        if l.startswith("ATOM"):
            key = (l[21], l[22:27])
            if key not in seen:
                seen.add(key)
                aa = THREE_TO_ONE.get(l[17:20].strip())
                if aa:
                    seq.append(aa)
    return "".join(seq)


def embed_sequence(seq: str, seed: int = 0) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    with torch.no_grad():
        h = model(**tok(seq, return_tensors="pt")).last_hidden_state[0, 1:-1].mean(0).numpy()  # drop BOS/EOS, mean-pool
    proj = np.random.default_rng(seed).normal(0, 1 / np.sqrt(h.size), (h.size, OUT_DIM))  # fixed JL-style projection
    z = h @ proj
    return (z / (np.linalg.norm(z) + 1e-8)).astype(np.float32)


def pocket_embedding(target_id: str, protein_pdb: str | Path | None = None, cache_dir: Path = ROOT / "data_cache" / "embeddings") -> np.ndarray:
    cache = cache_dir / f"{target_id}_pocket.npy"
    if cache.exists():
        return np.load(cache)
    if protein_pdb is None:
        from qumolbind.sim.target import find_target_files

        protein_pdb = find_target_files(target_id)[0]
    emb = embed_sequence(pocket_sequence(protein_pdb))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb
