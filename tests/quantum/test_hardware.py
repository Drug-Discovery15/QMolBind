import numpy as np
import pytest
import torch

from qumolbind.quantum.encoding import amplitude_state
from qumolbind.quantum.hardware import MAX_STATES, build_isa_pubs, run_dry, run_hardware
from qumolbind.quantum.torch_vqc import TorchVQC


@pytest.fixture(scope="module")
def small():
    n, K = 4, 4
    vqc = TorchVQC(n, 2, K, init_std=0.7)
    psi, _ = amplitude_state(torch.randn(3, 16, dtype=torch.float64), n)
    return vqc, psi.numpy(), n, K


def test_hardware_refuses_without_flag_and_token(small, monkeypatch) -> None:
    vqc, psi, n, K = small
    monkeypatch.delenv("IBM_QUANTUM_TOKEN", raising=False)
    with pytest.raises(PermissionError):
        run_hardware(vqc.theta.detach().numpy(), psi, n, K, allow_hardware=True)       # flag but no token
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "dummy")
    with pytest.raises(PermissionError):
        run_hardware(vqc.theta.detach().numpy(), psi, n, K, allow_hardware=False)      # token but no flag


def test_state_cap() -> None:
    with pytest.raises(ValueError):
        build_isa_pubs(np.zeros((1, 3, 1)), np.zeros((MAX_STATES + 1, 8)), 3, 3, None)


def test_transpile_to_heron_backend_exports_valid_pubs(small) -> None:
    from qiskit_ibm_runtime.fake_provider import FakeFez

    vqc, psi, n, K = small
    pubs, exact, cost = build_isa_pubs(vqc.theta.detach().numpy(), psi, n, K, FakeFez())
    assert len(pubs) == 3 and exact.shape == (3, K)
    isa, observables = pubs[0]
    assert set(isa.count_ops()) <= {"cz", "rz", "sx", "x", "barrier", "measure", "delay"}  # Heron native basis only
    assert cost["two_qubit_gates_mean"] > 0
    assert all(o.num_qubits == isa.num_qubits for o in observables)  # observables mapped onto the physical layout
    assert np.isfinite(exact).all()


def test_dry_run_is_offline_and_reports_noise_model(small) -> None:
    vqc, psi, n, K = small
    rep = run_dry(vqc.theta.detach().numpy(), psi, n, K, shots=1024)
    assert "NOT hardware" in rep.kind and "fake" in rep.backend
    assert np.isfinite(rep.mean_abs_error) and rep.mean_abs_error < 1.0 and rep.two_qubit_gates_mean > 0
    assert rep.job_ids == []  # nothing was submitted anywhere
