"""Gate-noise model for the VQC: exact density-matrix simulation in torch with depolarizing noise on 2-qubit gates.

Model (matches PennyLane ``default.mixed`` with ``qml.DepolarizingChannel(p)`` on BOTH wires after every CNOT/CZ):
    E_p(rho) = (1 - p) rho + p/3 (X rho X + Y rho Y + Z rho Z)   applied to each qubit the gate touched.
Single-qubit gates are noiseless (they are ~10x cleaner than 2-qubit gates on current devices). Evaluation-only
(no training through the noisy simulator); cost O(B 4^n) memory.
"""
from __future__ import annotations

import torch

from qumolbind.quantum.torch_vqc import CDTYPE, TorchVQC, apply_1q, bit, ry, rz, ring_pairs

_X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE)


def _conj_1q(rho: torch.Tensor, g: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """rho -> g rho g^dagger on qubit q for rho[B, N, N]."""
    B, N, _ = rho.shape
    rows = apply_1q(rho.transpose(1, 2).reshape(B * N, N), g, q, n)          # acts on the row index
    rho = rows.reshape(B, N, N).transpose(1, 2)
    cols = apply_1q(rho.reshape(B * N, N), g.conj(), q, n)                   # conj acts on the column index
    return cols.reshape(B, N, N)


def depolarize(rho: torch.Tensor, p: float, q: int, n: int) -> torch.Tensor:
    if p == 0.0:
        return rho
    return (1 - p) * rho + (p / 3) * (_conj_1q(rho, _X, q, n) + _conj_1q(rho, _Y, q, n) + _conj_1q(rho, _Z, q, n))


@torch.no_grad()
def noisy_expectations(vqc: TorchVQC, x: torch.Tensor, p: float) -> torch.Tensor:
    """<Z_i> (i<K, times scale) of the circuit with depolarizing noise ``p`` after each entangling gate."""
    n = vqc.n
    psi, _ = vqc.encode(x)
    rho = psi[:, :, None] * psi.conj()[:, None, :]
    N = 2**n
    idx = torch.arange(N)
    for l in range(vqc.L):
        for q in range(n):
            rho = _conj_1q(rho, ry(vqc.theta[l, q, 0]), q, n)
            if vqc.rotations == "ryrz":
                rho = _conj_1q(rho, rz(vqc.theta[l, q, 1]), q, n)
        if vqc.entangler == "none":
            continue
        for c, t in ring_pairs(n):
            if vqc.entangler == "cnot":
                g = torch.where(bit(idx, c, n) == 1, idx ^ (1 << (n - 1 - t)), idx)
                rho = rho[:, g][:, :, g]
            else:
                s = (1 - 2.0 * (bit(idx, c, n) & bit(idx, t, n)).to(torch.float64)).to(CDTYPE)
                rho = rho * s[:, None] * s[None, :]
            rho = depolarize(depolarize(rho, p, c, n), p, t, n)
    diag = torch.diagonal(rho, dim1=1, dim2=2).real
    out = diag @ vqc._zs[:, : vqc.K]
    return out * vqc.scale if vqc.scale is not None else out
