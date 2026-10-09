import numpy as np
import pennylane as qml
import pytest
import torch

from qumolbind.quantum.hw_cost import build_circuit, exact_expectations, hardware_cost_report, transpile_cost
from qumolbind.quantum.noise import noisy_expectations
from qumolbind.quantum.torch_vqc import TorchVQC, ring_pairs
from qumolbind.quantum.encoding import amplitude_state


@pytest.mark.parametrize("ent", ["cnot", "cz"])
@pytest.mark.parametrize("p", [0.0, 0.02, 0.1])
def test_noisy_sim_matches_pennylane_default_mixed(ent, p) -> None:
    n, L, K = 4, 2, 4
    vqc = TorchVQC(n, L, K, "ryrz", ent, init_std=1.0)
    x = torch.randn(2, 16, dtype=torch.float64)
    got = noisy_expectations(vqc, x, p)
    psi, _ = amplitude_state(x, n)
    th = vqc.theta.detach().numpy()

    @qml.qnode(qml.device("default.mixed", wires=n))
    def circ(state):
        qml.StatePrep(state, wires=range(n))
        for l in range(L):
            for q in range(n):
                qml.RY(th[l, q, 0], wires=q)
                qml.RZ(th[l, q, 1], wires=q)
            for c, t in ring_pairs(n):
                (qml.CNOT if ent == "cnot" else qml.CZ)(wires=[c, t])
                qml.DepolarizingChannel(p, wires=c)
                qml.DepolarizingChannel(p, wires=t)
        return [qml.expval(qml.PauliZ(i)) for i in range(K)]

    ref = np.array([[float(v) for v in circ(psi[i].numpy())] for i in range(2)])
    assert np.abs(ref - got.numpy()).max() < 1e-8


def test_noise_zero_equals_noiseless_and_noise_shrinks_signal() -> None:
    vqc = TorchVQC(6, 3, 6, init_std=1.0)
    x = torch.randn(3, 64, dtype=torch.float64)
    assert (noisy_expectations(vqc, x, 0.0) - vqc(x).detach()).abs().max() < 1e-10
    assert noisy_expectations(vqc, x, 0.05).abs().mean() < vqc(x).detach().abs().mean()


@pytest.mark.parametrize("rot,ent", [("ry", "cnot"), ("ryrz", "cz")])
def test_qiskit_export_matches_torch(rot, ent) -> None:
    n, L, K = 5, 3, 5
    vqc = TorchVQC(n, L, K, rot, ent, init_std=1.0)
    x = torch.randn(1, 32, dtype=torch.float64)
    psi, _ = amplitude_state(x, n)
    qc = build_circuit(vqc.theta.detach().numpy(), psi[0].numpy(), n, rot, ent)
    assert np.abs(exact_expectations(qc, K) - vqc(x).detach()[0].numpy()).max() < 1e-9


def test_transpile_to_heron_basis_and_cost_grows_with_n() -> None:
    rows = hardware_cost_report(ns=(3, 5), n_layers=2, n_states=1)
    assert rows[1]["cz_encoding_plus_ansatz"] > rows[0]["cz_encoding_plus_ansatz"]
    th = np.zeros((1, 4, 1))
    qc = build_circuit(th + 0.1, None, 4)
    from qiskit import transpile

    from qumolbind.quantum.hw_cost import HERON_BASIS, heron_like_coupling

    tq = transpile(qc, basis_gates=HERON_BASIS, coupling_map=heron_like_coupling(4), optimization_level=1, seed_transpiler=0)
    assert set(tq.count_ops()) <= set(HERON_BASIS) | {"barrier"}
    assert transpile_cost(qc, 4)["two_qubit_gates"] >= 3


def test_mps_exact_at_full_bond_dimension_and_degrades_for_small_chi() -> None:
    from qumolbind.quantum.mps import mps_expectations

    n, K = 6, 6
    vqc = TorchVQC(n, 3, K, init_std=0.8)
    x = torch.randn(1, 64, dtype=torch.float64)
    psi, _ = amplitude_state(x, n)
    exact = vqc(x).detach()[0].numpy()
    th = vqc.theta.detach().numpy()
    full = mps_expectations(th, psi[0].numpy(), n, K, chi=2 ** (n // 2))
    assert np.abs(full - exact).max() < 1e-8
    small = mps_expectations(th, psi[0].numpy(), n, K, chi=1)
    assert np.abs(small - exact).max() > 1e-3  # a product-state truncation cannot reproduce an entangled output
