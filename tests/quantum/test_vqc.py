import numpy as np
import pennylane as qml
import pytest
import torch

from qumolbind.quantum.encoding import amplitude_state, angle_state
from qumolbind.quantum.pl_vqc import PLVQC
from qumolbind.quantum.torch_vqc import TorchVQC, entanglement_entropy

torch.manual_seed(0)


def _rand_x(B: int, d: int = 256) -> torch.Tensor:
    return torch.randn(B, d, dtype=torch.float64)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_state_norm_after_encoding_and_ansatz(n) -> None:
    vqc = TorchVQC(n, 4, n, init_std=1.0)
    x = _rand_x(5)
    psi, norm = amplitude_state(x, n)
    assert torch.allclose(torch.linalg.vector_norm(psi, dim=-1), torch.ones(5, dtype=torch.float64), atol=1e-10)
    out = vqc.ansatz(psi)
    assert torch.allclose(torch.linalg.vector_norm(out, dim=-1), torch.ones(5, dtype=torch.float64), atol=1e-10)
    assert (norm > 0).all()  # input norm is logged but not used by the circuit


@pytest.mark.parametrize("rot,ent", [("ry", "cnot"), ("ry", "cz"), ("ryrz", "cnot"), ("ryrz", "cz"), ("ry", "none")])
def test_torch_matches_pennylane(rot, ent) -> None:
    n, L, K = 6, 3, 6
    vqc = TorchVQC(n, L, K, rot, ent, init_std=1.0)
    pl = PLVQC(n, L, K, rot, ent)
    x = _rand_x(7, 64)
    ref = pl(x, vqc.theta.detach())
    got = vqc(x).detach()
    assert (ref - got).abs().max() < 1e-6


def test_default_8_qubit_matches_pennylane() -> None:
    vqc = TorchVQC(8, 4, 8, init_std=0.5)
    pl = PLVQC(8, 4, 8)
    x = _rand_x(10)
    assert (pl(x, vqc.theta.detach()) - vqc(x).detach()).abs().max() < 1e-6


def test_pennylane_parameter_broadcasting_matches_loop() -> None:
    pl = PLVQC(4, 2, 4)
    theta = torch.randn(2, 4, 1, dtype=torch.float64)
    x = _rand_x(5, 16)
    batched = pl(x, theta)
    looped = torch.cat([pl(x[i : i + 1], theta) for i in range(5)])
    assert (batched - looped).abs().max() < 1e-12


def test_gradient_parity_autograd_vs_pennylane() -> None:
    n, L, K = 4, 2, 4
    x = _rand_x(3, 16)
    w = torch.randn(3, K, dtype=torch.float64)
    vqc = TorchVQC(n, L, K, "ryrz", "cnot", init_std=0.7)
    (vqc(x) * w).sum().backward()
    g_torch = vqc.theta.grad.clone()
    for dm in ("backprop", "parameter-shift"):
        pl = PLVQC(n, L, K, "ryrz", "cnot", diff_method=dm)
        th = vqc.theta.detach().clone().requires_grad_(True)
        (pl(x, th) * w).sum().backward()
        assert (th.grad - g_torch).abs().max() < 1e-6, dm


def test_zero_vector_input_no_nan() -> None:
    vqc = TorchVQC(8, 4, 8)
    out = vqc(torch.zeros(2, 256))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(vqc.theta.grad).all()
    pl = PLVQC(8, 4, 8)
    assert torch.isfinite(pl(torch.zeros(2, 256), vqc.theta.detach())).all()


def test_shot_estimates_converge_at_inverse_sqrt_rate() -> None:
    vqc = TorchVQC(6, 3, 6, init_std=1.0)
    x = _rand_x(1, 64)
    exact = vqc(x).detach()
    g = torch.Generator().manual_seed(0)
    rms = {}
    for shots in (64, 256, 1024, 4096):
        errs = torch.stack([vqc(x, shots=shots, generator=g).detach() - exact for _ in range(300)])
        rms[shots] = float(errs.pow(2).mean().sqrt())
    slope = np.polyfit(np.log([64, 256, 1024, 4096]), np.log(list(rms.values())), 1)[0]
    assert -0.6 < slope < -0.4, (slope, rms)
    assert rms[4096] < 1.2 / np.sqrt(4096)  # binomial bound: Var(<Z>_hat) <= 1/shots


def test_shot_sampling_keeps_analytic_gradient() -> None:
    vqc = TorchVQC(4, 2, 4)
    vqc(_rand_x(2, 16), shots=128).sum().backward()
    assert vqc.theta.grad.abs().sum() > 0


def test_angle_encoding_norm_and_pennylane_agreement() -> None:
    a = torch.randn(3, 6, dtype=torch.float64)
    psi = angle_state(a, 6)
    assert torch.allclose(torch.linalg.vector_norm(psi, dim=-1), torch.ones(3, dtype=torch.float64), atol=1e-12)

    @qml.qnode(qml.device("default.qubit", wires=6))
    def circ(an):
        for q in range(6):
            qml.RY(an[q], wires=q)
        return qml.state()

    for i in range(3):
        assert np.allclose(circ(a[i].numpy()), psi[i].numpy(), atol=1e-12)


def test_no_entanglement_gives_product_state() -> None:
    vqc = TorchVQC(6, 3, 6, entangler="none", encoding="angle", init_std=1.0)
    x = torch.randn(4, 6, dtype=torch.float64)
    assert entanglement_entropy(vqc.final_state(x), 6).abs().max() < 1e-6


def test_param_counts() -> None:
    assert TorchVQC(8, 4, 8).n_quantum_params() == 32
    assert TorchVQC(8, 4, 8, "ryrz").n_quantum_params() == 64
    v = TorchVQC(8, 4, 8, trainable_scale=True, input_projection=True)
    assert v.n_classical_params() == 8 + 256 * 256 + 256
