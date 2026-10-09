"""Offline tests: everything here uses tests/fixtures only."""
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from qumolbind.data import bindingdb
from qumolbind.data.fixtures import FIXTURE_DIR
from qumolbind.data.pdb import parse_hetero_groups
from qumolbind.data.zinc import load_tranche


def test_fixture_pocket_has_single_ligand() -> None:
    text = (FIXTURE_DIR / "1CIL_pocket.pdb").read_text()
    groups = parse_hetero_groups(text)
    assert {k[0] for k in groups} == {"ETS"}
    assert any(l.startswith("ATOM") for l in text.splitlines())


def test_rotor_count_ets() -> None:
    mol = Chem.MolFromSmiles("CCOc1ccc2nc(sc2c1)S(N)(=O)=O")  # ethoxzolamide
    assert rdMolDescriptors.CalcNumRotatableBonds(mol) == 3


def test_p_affinity() -> None:
    assert bindingdb.p_affinity(1.0) == pytest.approx(9.0)
    assert bindingdb.p_affinity(1000.0) == pytest.approx(6.0)


def test_normalize_dedupes_and_drops_censored() -> None:
    recs = [
        {"smile": "c1ccccc1O", "affinity_type": "Ki", "affinity": "100", "monomerid": "1"},
        {"smile": "Oc1ccccc1", "affinity_type": "IC50", "affinity": "10", "monomerid": "1"},  # same mol, lower-priority type
        {"smile": "CCO", "affinity_type": "Kd", "affinity": ">10000", "monomerid": "2"},      # censored
        {"smile": "not_a_smiles", "affinity_type": "Ki", "affinity": "5", "monomerid": "3"},
    ]
    df = bindingdb.normalize(recs)
    assert len(df) == 1
    row = df.iloc[0]
    assert row.measurement == "Ki" and row.n_measurements == 2
    assert row.p_affinity == pytest.approx(7.0)


def test_normalize_fixture() -> None:
    recs = json.loads((FIXTURE_DIR / "bindingdb_P00918_tiny.json").read_text())
    df = bindingdb.normalize(recs)
    assert len(df) > 10 and df.smiles.is_unique
    assert set(df.measurement) <= {"Ki", "Kd", "IC50"}


def test_zinc_loader_local(tmp_path: Path) -> None:
    f = tmp_path / "t.smi"
    f.write_text("smiles zinc_id\nCCO ZINC1\nCCN ZINC2\nCCC ZINC3\n")
    df = load_tranche(f, max_molecules=2)
    assert list(df.zinc_id) == ["ZINC1", "ZINC2"]
