"""Symmetry-corrected heavy-atom RMSD to the native pose (evaluation only; never used in state/reward)."""
from __future__ import annotations

import numpy as np
from rdkit import Chem


class SymmetryRMSD:
    """In-place (no alignment) RMSD minimised over graph automorphisms of the heavy-atom molecule.

    The pose lives in the fixed protein frame, so no superposition is performed (same convention as
    RDKit ``rdMolAlign.CalcRMS``, which this is tested against).
    """

    def __init__(self, mol: Chem.Mol, native_coords: np.ndarray, max_matches: int = 20000) -> None:
        heavy = Chem.RWMol(Chem.RemoveHs(mol))
        # topology-only automorphisms: carboxylate / nitro / sulfonyl oxygens are crystallographically
        # indistinguishable, so ignore bond orders, aromaticity and formal charges when matching.
        for a in heavy.GetAtoms():
            a.SetFormalCharge(0)
            a.SetIsAromatic(False)
            a.SetNoImplicit(True)
        for b in heavy.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        self.n_heavy = heavy.GetNumAtoms()
        matches = heavy.GetSubstructMatches(heavy, uniquify=False, useChirality=False, maxMatches=max_matches)
        self.perms = np.array(matches, dtype=np.int64)  # (n_perm, n_heavy)
        self.native_heavy = np.asarray(native_coords, dtype=np.float64)[: self.n_heavy]

    def __call__(self, coords: np.ndarray) -> float:
        pose = np.asarray(coords, dtype=np.float64)[: self.n_heavy]
        # match query atom j -> pose atom perms[p, j]
        d2 = np.sum((pose[self.perms] - self.native_heavy[None]) ** 2, axis=-1).mean(axis=-1)
        return float(np.sqrt(d2.min()))
