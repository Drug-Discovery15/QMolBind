"""Input encodings for the VQC.

Amplitude encoding: x -> x / (||x|| + 1e-8) loaded as the amplitudes of an n-qubit state (2^n features).
  * The normalisation DISCARDS ||x|| (only the direction of x reaches the circuit); the norm is returned as a diagnostic.
  * A (near-)zero vector would give an un-normalised all-zero state; it is mapped to the basis state |0...0> instead
    (documented convention, so the state norm is always 1 and nothing is NaN).
  * If x has a different length than 2^n it is zero-padded (shorter) or truncated (longer).
Angle encoding: features -> RY(angle) rotations on n qubits (one feature per qubit).
"""
from __future__ import annotations

import torch

EPS = 1e-8


def fit_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Zero-pad / truncate the last axis to ``dim``."""
    d = x.shape[-1]
    if d == dim:
        return x
    if d < dim:
        return torch.nn.functional.pad(x, (0, dim - d))
    return x[..., :dim]


def amplitude_state(x: torch.Tensor, n_qubits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x[B,d] -> (psi[B,2^n] complex128 normalised, norm[B] of the (fitted) input)."""
    x = fit_to_dim(x.to(torch.float64), 2**n_qubits)
    norm = torch.linalg.vector_norm(x, dim=-1)
    psi = x / (norm[..., None] + EPS)
    zero = norm < EPS
    if zero.any():
        e0 = torch.zeros_like(psi)
        e0[..., 0] = 1.0
        psi = torch.where(zero[..., None], e0, psi)
    # exact renormalisation (removes the O(EPS) deficit of x/(|x|+EPS))
    psi = psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True)
    return psi.to(torch.complex128), norm


def angle_state(angles: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Product state prod_q RY(angles[q])|0> as a [B, 2^n] complex128 vector (qubit 0 = most significant bit)."""
    a = fit_to_dim(angles.to(torch.float64), n_qubits)
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    psi = torch.ones(a.shape[0], 1, dtype=torch.float64)
    for q in range(n_qubits):
        qubit = torch.stack([c[:, q], s[:, q]], dim=-1)           # [B,2]
        psi = (psi[:, :, None] * qubit[:, None, :]).reshape(a.shape[0], -1)
    return psi.to(torch.complex128)
