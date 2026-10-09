"""State vector shared by ALL policies (MLP, VQC, search baselines' surrogate/AL features).

Layout (fixed, target independent; K_MAX = 8 torsion slots, missing torsions are zero):
  [ sin(tors) (8) | cos(tors) (8) | normalised energy terms vdW,elec,solv (3) | t/T (1)
    | ligand GNN embedding (64, optional) | pocket ESM embedding (32, optional) ]  zero-padded to d = 256.
RMSD and native coordinates NEVER enter the state (label leakage).
"""
from __future__ import annotations

import numpy as np

K_MAX = 8
LIG_EMB_DIM = 64
POCKET_EMB_DIM = 32
OFF_TERMS = 2 * K_MAX
OFF_T = OFF_TERMS + 3
OFF_LIG = OFF_T + 1
OFF_POCKET = OFF_LIG + LIG_EMB_DIM


def symlog(x: np.ndarray | float, scale: float = 100.0) -> np.ndarray | float:
    """Signed log compression: ~linear for |x| << scale, logarithmic for clash-scale energies."""
    return np.sign(x) * np.log1p(np.abs(x) / scale)


def build_state(
    torsions_deg: np.ndarray,
    terms: np.ndarray,
    t: int,
    horizon: int,
    dim: int = 256,
    lig_emb: np.ndarray | None = None,
    pocket_emb: np.ndarray | None = None,
    energy_scale: float = 100.0,
) -> np.ndarray:
    s = np.zeros(dim, dtype=np.float32)
    k = len(torsions_deg)
    rad = np.radians(torsions_deg)
    s[:k] = np.sin(rad)
    s[K_MAX : K_MAX + k] = np.cos(rad)
    s[OFF_TERMS : OFF_TERMS + 3] = symlog(np.asarray(terms), energy_scale) / 5.0
    s[OFF_T] = t / horizon
    if lig_emb is not None:
        s[OFF_LIG : OFF_LIG + LIG_EMB_DIM] = lig_emb
    if pocket_emb is not None:
        s[OFF_POCKET : OFF_POCKET + POCKET_EMB_DIM] = pocket_emb
    return s
