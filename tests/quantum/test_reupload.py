import numpy as np
import pytest
import torch

from qumolbind.quantum.hw_cost import build_circuit, exact_expectations
from qumolbind.quantum.mps import mps_expectations_reupload
from qumolbind.quantum.counts import vqc_param_count
from qumolbind.quantum.torch_vqc import TorchVQC


def _vqc(**kw):
    torch.manual_seed(0)
    return TorchVQC(6, 3, 6, encoding="reupload", equator_init=True, readout_affine=True, feat_dim=10, init_std=0.5, reupload_std=0.7, **kw)


@pytest.mark.parametrize("rot,ent", [("ry", "cnot"), ("ryrz", "cz")])
def test_reupload_matches_independent_qiskit_circuit(rot, ent) -> None:
    v = _vqc(rotations=rot, entangler=ent)
    with torch.no_grad():
        v.r_scale.copy_(torch.linspace(0.5, 1.5, 6)); v.r_bias.copy_(torch.linspace(-0.2, 0.2, 6))
    x = torch.randn(3, 20, dtype=torch.float64)
    got = v(x).detach().numpy()
    for i in range(3):
        angles = v.theta.detach().numpy() + v.reupload_angles(x[i : i + 1])[0].detach().numpy()
        z = exact_expectations(build_circuit(angles, None, 6, rot, ent, equator_init=True), 6)
        assert np.abs(z * v.r_scale.detach().numpy() + v.r_bias.detach().numpy() - got[i]).max() < 1e-9


def test_equator_start_has_zero_means_and_finite_gradients() -> None:
    v = TorchVQC(8, 4, 8, encoding="reupload", equator_init=True, readout_affine=True, reupload_std=0.0, init_std=0.0)
    assert v(torch.randn(4, 256, dtype=torch.float64)).abs().max() < 1e-12      # neutral policy at init
    v2 = _vqc()
    out = v2(torch.randn(5, 20, dtype=torch.float64)); out.pow(2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in v2.parameters())


def test_gradient_matches_finite_differences() -> None:
    v = _vqc()
    x = torch.randn(2, 20, dtype=torch.float64)
    w = torch.randn(2, 6, dtype=torch.float64)
    (v(x) * w).sum().backward()
    g = v.theta.grad.clone()
    idx, eps = (1, 2, 0), 1e-6
    with torch.no_grad():
        v.theta[idx] += eps; hi = (v(x) * w).sum(); v.theta[idx] -= 2 * eps; lo = (v(x) * w).sum(); v.theta[idx] += eps
    assert abs(float((hi - lo) / (2 * eps)) - float(g[idx])) < 1e-6


def test_mps_exact_at_full_chi_for_reupload() -> None:
    v = _vqc()
    x = torch.randn(1, 20, dtype=torch.float64)
    assert np.abs(mps_expectations_reupload(v, x[0].numpy(), chi=8) - v(x).detach()[0].numpy()).max() < 1e-8
    assert np.abs(mps_expectations_reupload(v, x[0].numpy(), chi=1) - v(x).detach()[0].numpy()).max() > 1e-4


def test_param_count_bookkeeping_matches_model() -> None:
    v = TorchVQC(8, 4, 8, encoding="reupload", equator_init=True, readout_affine=True)
    c = vqc_param_count(8, 4, "ry", 8, encoding="reupload", readout_affine=True)
    assert c["quantum"] == v.n_quantum_params() and c["classical"] == v.n_classical_params()
