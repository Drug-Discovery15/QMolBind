from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms

from qumolbind.chem.ligand import LigandModel
from qumolbind.chem.rmsd import SymmetryRMSD

FIX = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def native_model() -> LigandModel:
    mol = Chem.MolFromMolFile(str(FIX / "1cil_ligand_native.sdf"), removeHs=False)
    return LigandModel(mol)


@pytest.fixture(scope="module")
def flexible_model() -> LigandModel:
    mol = Chem.AddHs(Chem.MolFromSmiles("CCOCCN(C)c1ccc(cc1)C(=O)OCC"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return LigandModel(mol)


def test_torsions_found(native_model: LigandModel) -> None:
    assert 3 <= native_model.K <= 8


@pytest.mark.parametrize("model_name", ["native_model", "flexible_model"])
def test_plus_minus_restores(model_name: str, request) -> None:
    lm: LigandModel = request.getfixturevalue(model_name)
    rng = np.random.default_rng(0)
    x = lm.native
    for _ in range(20):
        d = rng.uniform(-180, 180, lm.K)
        y = lm.apply_torsion_deltas(lm.apply_torsion_deltas(x, d), -d)
        assert np.abs(y - x).max() <= 1e-6


def test_root_fixed_and_dihedral_changes_exactly(flexible_model: LigandModel) -> None:
    lm = flexible_model
    rng = np.random.default_rng(1)
    d = rng.uniform(-90, 90, lm.K)
    before = lm.get_torsions_deg(lm.native)
    y = lm.apply_torsion_deltas(lm.native, d)
    assert np.abs(y[lm.root_atoms] - lm.native[lm.root_atoms]).max() < 1e-9
    after = lm.get_torsions_deg(y)
    diff = (after - before - d + 180) % 360 - 180
    assert np.abs(diff).max() < 1e-6


def test_matches_rdkit_dihedral(flexible_model: LigandModel) -> None:
    lm = flexible_model
    y = lm.apply_torsion_deltas(lm.native, np.full(lm.K, 37.0))
    conf = Chem.Conformer(lm.mol.GetConformer())
    for i, p in enumerate(y):
        conf.SetAtomPosition(i, p.tolist())
    for k, t in enumerate(lm.torsions):
        rd = rdMolTransforms.GetDihedralDeg(conf, int(t.i), int(t.a), int(t.b), int(t.l))
        assert abs((rd - lm.get_torsions_deg(y)[k] + 180) % 360 - 180) < 1e-6


def test_bond_lengths_preserved(flexible_model: LigandModel) -> None:
    lm = flexible_model
    y = lm.apply_torsion_deltas(lm.native, np.random.default_rng(2).uniform(-180, 180, lm.K))
    for b in lm.mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        assert abs(np.linalg.norm(y[i] - y[j]) - np.linalg.norm(lm.native[i] - lm.native[j])) < 1e-9


def test_rmsd_symmetry_and_rdkit_agreement() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)[O-]"))  # symmetric phenyl + carboxylate
    AllChem.EmbedMolecule(mol, randomSeed=3)
    lm = LigandModel(mol)
    rm = SymmetryRMSD(mol, lm.native)
    assert rm(lm.native) == 0.0
    # swap the two equivalent carboxylate oxygens: symmetry-corrected RMSD must stay ~0
    heavy = Chem.RemoveHs(mol)
    o = [a.GetIdx() for a in heavy.GetAtoms() if a.GetSymbol() == "O"]
    swapped = lm.native.copy()
    swapped[[o[0], o[1]]] = swapped[[o[1], o[0]]]
    assert rm(swapped) < 1e-9
    # compare to RDKit CalcRMS on a displaced pose (molecule without extra topological symmetry vs. RDKit's typed matching)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    lm = LigandModel(mol)
    rm = SymmetryRMSD(mol, lm.native)
    pose = lm.native + np.random.default_rng(0).normal(0, 0.3, lm.native.shape)
    ref, prb = Chem.RemoveHs(mol), Chem.RemoveHs(mol)
    for m, c in ((ref, lm.native), (prb, pose)):
        conf = m.GetConformer()
        for i in range(m.GetNumAtoms()):
            conf.SetAtomPosition(i, c[i].tolist())
    assert rm(pose) == pytest.approx(rdMolAlign.CalcRMS(prb, ref), abs=1e-6)


def test_protein_fixture_is_hydrogenated_and_capped() -> None:
    text = (FIX / "1cil_protein.pdb").read_text()
    assert " ACE " in text and " NME " in text and " ZN " in text
    assert sum(1 for l in text.splitlines() if l.startswith("ATOM") and l[76:78].strip() == "H") > 500
