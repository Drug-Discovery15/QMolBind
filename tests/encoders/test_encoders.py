import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from qumolbind.encoders.affinity_train import ecfp4, scaffold_split
from qumolbind.encoders.esm import pocket_sequence
from qumolbind.encoders.gnn import AffinityGNN, radius_graph_torch, smiles_to_data, triplets_torch

SMILES = ["CCOc1ccc2nc(sc2c1)S(N)(=O)=O", "CC(=O)Nc1ccc(O)cc1", "c1ccccc1C(=O)O", "CCN(CC)CCNC(=O)c1ccc(N)cc1"]


@pytest.fixture(scope="module")
def batch():
    return Batch.from_data_list([smiles_to_data(s, 5.0) for s in SMILES])


@pytest.mark.parametrize("kind", ["dimenet", "schnet", "gine"])
def test_gnn_shapes_and_grad(kind, batch) -> None:
    m = AffinityGNN(kind, 32)
    assert m.embed(batch).shape == (4, 64)
    out = m(batch)
    assert out.shape == (4,) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad and p.grad is not None)


def test_triplets_match_brute_force(batch) -> None:
    ei = radius_graph_torch(batch.pos, 5.0, batch.batch)
    _, _, ii, jj, kk, kj, ji = triplets_torch(ei, batch.pos.size(0))
    R, C = ei
    brute = {(int(f), int(e)) for e in range(R.numel()) for f in range(R.numel()) if C[f] == R[e] and R[f] != C[e]}
    assert brute == set(zip(kj.tolist(), ji.tolist()))


def test_radius_graph_respects_batches_and_cutoff(batch) -> None:
    ei = radius_graph_torch(batch.pos, 3.0, batch.batch)
    assert (batch.batch[ei[0]] == batch.batch[ei[1]]).all()
    assert ((batch.pos[ei[0]] - batch.pos[ei[1]]).norm(dim=1) <= 3.0 + 1e-6).all()


def test_embedding_is_deterministic_and_rotation_invariant() -> None:
    m = AffinityGNN("dimenet", 32).eval()
    d = smiles_to_data(SMILES[0], 1.0)
    rot = torch.linalg.qr(torch.randn(3, 3)).Q
    d2 = d.clone(); d2.pos = d.pos @ rot
    with torch.no_grad():
        a, b = m.embed(Batch.from_data_list([d])), m.embed(Batch.from_data_list([d2]))
    assert torch.allclose(a, b, atol=1e-4)


def test_scaffold_split_disjoint() -> None:
    smi = SMILES * 10
    tr, va, te = scaffold_split(smi, seed=0)
    assert not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(va) & set(te))
    from rdkit.Chem.Scaffolds import MurckoScaffold

    sc = lambda idx: {MurckoScaffold.MurckoScaffoldSmiles(smiles=smi[i]) for i in idx}
    assert not (sc(tr) & sc(te))


def test_ecfp_shape() -> None:
    X = ecfp4(SMILES)
    assert X.shape == (4, 2048) and X.sum() > 0


def test_pocket_sequence_from_fixture() -> None:
    from pathlib import Path

    seq = pocket_sequence(Path(__file__).resolve().parents[1] / "fixtures" / "1cil_protein.pdb")
    assert len(seq) > 50 and set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
