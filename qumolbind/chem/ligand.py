"""Ligand torsion model: rotatable bonds, torsion read/write, deltas with a fixed root fragment.

All geometry is Cartesian (N,3) float64 numpy arrays in the (fixed) protein frame.
Torsions are applied by rigid rotation of the moving side about the bond axis (Rodrigues); this is
numerically identical to RDKit's SetDihedral* but vectorised/cheap, and is cross-checked against
``rdMolTransforms.GetDihedralDeg`` in tests (see DECISIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

# RDKit "strict" rotatable bond definition (excludes amides, terminal CF3/tBu, etc.)
_STRICT = Chem.MolFromSmarts(
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])"
    "&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])"
    "&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@"
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
)


@dataclass(frozen=True)
class Torsion:
    a: int          # root-side bond atom
    b: int          # moving-side bond atom (axis a->b)
    i: int          # reference neighbour of a (defines dihedral)
    l: int          # reference neighbour of b
    moving: np.ndarray  # indices rotated when this torsion changes (includes b, and hydrogens)


def dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _rotate(coords: np.ndarray, idx: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> None:
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    v = coords[idx] - origin
    coords[idx] = origin + v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1 - c)


class LigandModel:
    """Native pose + torsion definitions for a hydrogen-complete ligand."""

    def __init__(self, mol: Chem.Mol, max_torsions: int = 8) -> None:
        self.mol = Chem.Mol(mol)
        self.native = self.mol.GetConformer().GetPositions().astype(np.float64)
        self.n_atoms = self.mol.GetNumAtoms()
        self.heavy_idx = np.array([a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1])
        self.n_rotatable_total = 0
        self.torsions: list[Torsion] = self._build(max_torsions)
        self.K = len(self.torsions)

    # ---- construction -------------------------------------------------
    def _build(self, max_torsions: int) -> list[Torsion]:
        heavy = Chem.RemoveHs(self.mol)  # heavy atoms keep indices 0..n_heavy-1 (Hs are appended by AddHs)
        bonds = [tuple(m) for m in heavy.GetSubstructMatches(_STRICT)]
        self.n_rotatable_total = len(bonds)
        # root fragment: largest heavy-atom connected component after cutting rotatable bonds
        rw = Chem.RWMol(heavy)
        for u, v in bonds:
            rw.RemoveBond(u, v)
        frags = Chem.GetMolFrags(rw, asMols=False)
        root = set(max(frags, key=len))
        self.root_atoms = np.array(sorted(root))
        adj = [[n.GetIdx() for n in a.GetNeighbors()] for a in self.mol.GetAtoms()]  # full graph incl. H

        def side(start: int, blocked: int) -> set[int]:
            seen, stack = {start}, [start]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y != blocked and y not in seen:
                        seen.add(y)
                        stack.append(y)
            return seen

        cands = []
        root_ref = next(iter(root))
        for u, v in bonds:
            su = side(u, v)
            near_root, far = (u, v) if root_ref in su else (v, u)
            moving = side(far, near_root)
            if root_ref in moving:  # cyclic through other path (should not happen for non-ring bonds)
                continue
            ref_i = next(n for n in adj[near_root] if n != far and self.mol.GetAtomWithIdx(n).GetAtomicNum() > 1)
            ref_l = next((n for n in adj[far] if n != near_root and self.mol.GetAtomWithIdx(n).GetAtomicNum() > 1), None)
            if ref_l is None:
                continue
            cands.append(Torsion(near_root, far, ref_i, ref_l, np.array(sorted(moving))))
        # cap at K: keep the torsions that move the most atoms (largest effect on pose), then order root->leaf
        if len(cands) > max_torsions:
            cands = sorted(cands, key=lambda t: -len(t.moving))[:max_torsions]
        # order so inner torsions (bigger moving sets) come first
        return sorted(cands, key=lambda t: -len(t.moving))

    # ---- geometry -----------------------------------------------------
    def get_torsions_deg(self, coords: np.ndarray) -> np.ndarray:
        return np.array([dihedral_deg(coords[t.i], coords[t.a], coords[t.b], coords[t.l]) for t in self.torsions])

    def apply_torsion_deltas(self, coords: np.ndarray, deltas_deg: np.ndarray) -> np.ndarray:
        """Return new coords with each torsion changed by ``deltas_deg[k]``; root fragment stays fixed."""
        out = np.array(coords, dtype=np.float64, copy=True)
        for t, d in zip(self.torsions, np.asarray(deltas_deg, dtype=np.float64)):
            if d == 0.0:
                continue
            _rotate(out, t.moving, out[t.a], out[t.b] - out[t.a], np.radians(d))
        return out

    def set_torsions_deg(self, coords: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
        cur = self.get_torsions_deg(coords)
        return self.apply_torsion_deltas(coords, np.asarray(target_deg) - cur)

    def randomize(self, rng: np.random.Generator, coords: np.ndarray | None = None) -> np.ndarray:
        """Uniformly random absolute torsions in [-180, 180)."""
        base = self.native if coords is None else coords
        return self.set_torsions_deg(base, rng.uniform(-180.0, 180.0, size=self.K))

    def describe(self) -> str:
        lines = [f"K={self.K} torsions (of {self.n_rotatable_total} rotatable bonds), root atoms={len(self.root_atoms)}"]
        for k, t in enumerate(self.torsions):
            sym = lambda x: self.mol.GetAtomWithIdx(int(x)).GetSymbol() + str(int(x))
            lines.append(f"  [{k}] {sym(t.i)}-{sym(t.a)}-{sym(t.b)}-{sym(t.l)}  moves {len(t.moving)} atoms  native={self.get_torsions_deg(self.native)[k]:.1f} deg")
        return "\n".join(lines)
