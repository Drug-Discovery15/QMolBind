"""Ligand parametrization without OpenFF/AmberTools (not pip-installable on Windows; see DECISIONS.md D3/D11).

Force field used for the ligand (all of it generated here):
  * partial charges: MMFF94 (RDKit); fallback Gasteiger (net charge corrected) when MMFF typing fails
  * Lennard-Jones: UFF per-element x_ii, D_ii (RDKit) -> sigma = x/2^(1/6), eps = D; polar H (on N/O/S) get eps = 0 as in AMBER-type FFs
  * bonded: harmonic bonds/angles fixed at the native geometry with generic force constants.
    Torsion moves never change bond lengths/angles, so these terms are pose-independent and cancel in E_int.
  * torsional strain / intramolecular sterics: MMFF94 intramolecular energy from RDKit (``StrainEnergy``)
Exposed to OpenMM as an XML ForceField fragment (one atom type per ligand atom) so the standard
``ForceField(..., 'implicit/gbn2.xml', ligand_xml)`` pipeline (GBn2 radii from the topology) applies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openmm import app
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import rdPartialCharges

KCAL_TO_KJ = 4.184
RESNAME = "LIG"


@dataclass
class LigandParams:
    charges: np.ndarray       # e
    sigma_nm: np.ndarray
    eps_kj: np.ndarray
    charge_method: str        # 'mmff94' | 'gasteiger'


def parametrize_ligand(mol: Chem.Mol) -> LigandParams:
    n = mol.GetNumAtoms()
    net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    method = "mmff94"
    q = np.zeros(n)
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol) if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol) else None
    if props is not None:
        q = np.array([props.GetMMFFPartialCharge(i) for i in range(n)])
    else:
        method = "gasteiger"
        m2 = Chem.Mol(mol)
        rdPartialCharges.ComputeGasteigerCharges(m2)
        q = np.array([float(a.GetProp("_GasteigerCharge")) for a in m2.GetAtoms()])
        if not np.all(np.isfinite(q)):
            raise ValueError("charge assignment failed (non-finite Gasteiger charges)")
    q += (net - q.sum()) / n  # enforce exact net formal charge
    sigma = np.zeros(n)
    eps = np.zeros(n)
    for i, a in enumerate(mol.GetAtoms()):
        pr = rdForceFieldHelpers.GetUFFVdWParams(mol, i, i)  # (x Angstrom, D kcal/mol)
        if pr is None:
            raise ValueError(f"no UFF vdW parameters for atom {a.GetSymbol()}")
        x, d = pr
        sigma[i] = x / 2 ** (1 / 6) / 10.0
        eps[i] = d * KCAL_TO_KJ
        if a.GetAtomicNum() == 1 and any(nb.GetAtomicNum() in (7, 8, 16) for nb in a.GetNeighbors()):
            sigma[i], eps[i] = 0.1, 0.0
    return LigandParams(q, sigma, eps, method)


def _atom_names(mol: Chem.Mol) -> list[str]:
    return [f"{a.GetSymbol().upper()}{i}" for i, a in enumerate(mol.GetAtoms())]


def ligand_topology(mol: Chem.Mol, chain_id: str = "L") -> app.Topology:
    top = app.Topology()
    chain = top.addChain(chain_id)
    res = top.addResidue(RESNAME, chain)
    names = _atom_names(mol)
    atoms = [top.addAtom(names[i], app.Element.getByAtomicNumber(a.GetAtomicNum()), res) for i, a in enumerate(mol.GetAtoms())]
    for b in mol.GetBonds():
        top.addBond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    return top


def ligand_forcefield_xml(mol: Chem.Mol, params: LigandParams, coords_angstrom: np.ndarray) -> str:
    n = mol.GetNumAtoms()
    names = _atom_names(mol)
    types = [f"lig-{i}" for i in range(n)]
    xyz = np.asarray(coords_angstrom) / 10.0
    pt = Chem.GetPeriodicTable()
    out = ["<ForceField>", " <AtomTypes>"]
    for i, a in enumerate(mol.GetAtoms()):
        out.append(f'  <Type name="{types[i]}" class="{types[i]}" element="{a.GetSymbol()}" mass="{pt.GetAtomicWeight(a.GetAtomicNum()):.4f}"/>')
    out += [" </AtomTypes>", f' <Residues><Residue name="{RESNAME}">']
    for i in range(n):
        out.append(f'  <Atom name="{names[i]}" type="{types[i]}"/>')
    for b in mol.GetBonds():
        out.append(f'  <Bond atomName1="{names[b.GetBeginAtomIdx()]}" atomName2="{names[b.GetEndAtomIdx()]}"/>')
    out += [" </Residue></Residues>", " <HarmonicBondForce>"]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        out.append(f'  <Bond class1="{types[i]}" class2="{types[j]}" length="{np.linalg.norm(xyz[i] - xyz[j]):.6f}" k="250000.0"/>')
    out += [" </HarmonicBondForce>", " <HarmonicAngleForce>"]
    for c in mol.GetAtoms():
        nb = [x.GetIdx() for x in c.GetNeighbors()]
        for a_i in range(len(nb)):
            for a_j in range(a_i + 1, len(nb)):
                i, k, j = nb[a_i], c.GetIdx(), nb[a_j]
                v1, v2 = xyz[i] - xyz[k], xyz[j] - xyz[k]
                ang = np.arccos(np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))
                out.append(f'  <Angle class1="{types[i]}" class2="{types[k]}" class3="{types[j]}" angle="{ang:.6f}" k="400.0"/>')
    out += [" </HarmonicAngleForce>", ' <NonbondedForce coulomb14scale="0.833333333333" lj14scale="0.5">']
    for i in range(n):
        out.append(f'  <Atom type="{types[i]}" charge="{params.charges[i]:.6f}" sigma="{params.sigma_nm[i]:.6f}" epsilon="{params.eps_kj[i]:.6f}"/>')
    out += [" </NonbondedForce>", "</ForceField>"]
    return "\n".join(out)


class StrainEnergy:
    """MMFF94 intramolecular energy (kJ/mol) of a pose, relative to the native pose.

    Bonds/angles are identical for all torsion-space poses, so differences come from torsions,
    1-4 and non-bonded intramolecular terms (i.e. ligand self-clashes and eclipsed torsions).
    """

    def __init__(self, mol: Chem.Mol, native_coords: np.ndarray) -> None:
        self.mol = Chem.Mol(mol)
        self.ok = rdForceFieldHelpers.MMFFHasAllMoleculeParams(self.mol)
        self.n = mol.GetNumAtoms()
        if self.ok:
            self.props = rdForceFieldHelpers.MMFFGetMoleculeProperties(self.mol)
            self.ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(self.mol, self.props)
        else:  # UFF fallback keeps a (cruder) strain term rather than silently dropping it
            self.ff = rdForceFieldHelpers.UFFGetMoleculeForceField(self.mol)
        self.e0 = self._raw(native_coords)

    def _raw(self, coords: np.ndarray) -> float:
        return float(self.ff.CalcEnergy(np.asarray(coords, dtype=np.float64).ravel().tolist())) * KCAL_TO_KJ

    def __call__(self, coords: np.ndarray) -> float:
        return self._raw(coords) - self.e0
