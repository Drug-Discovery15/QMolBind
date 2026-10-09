"""Batched pure-PyTorch statevector simulator for the QPPO actor (complex128, exact autograd).

Convention (same as PennyLane): qubit 0 is the MOST significant bit of the basis-state index.
Single-qubit gates act by reshaping the state to [B, 2^q, 2, 2^(n-q-1)]; CNOT/CZ rings are permutations / sign vectors
over the basis index (precomputed once per (n, entangler)).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from qumolbind.quantum.encoding import amplitude_state, angle_state, fit_to_dim

CDTYPE = torch.complex128


def bit(i: torch.Tensor, q: int, n: int) -> torch.Tensor:
    return (i >> (n - 1 - q)) & 1


def ring_pairs(n: int) -> list[tuple[int, int]]:
    """CNOT/CZ ring: (0,1),(1,2),...,(n-1,0). For n == 2 the ring degenerates to one pair."""
    if n == 2:
        return [(0, 1)]
    return [(q, (q + 1) % n) for q in range(n)]


def cnot_ring_permutation(n: int) -> torch.Tensor:
    """perm such that (U psi)[i] = psi[perm[i]] for the sequence CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0) (applied in order)."""
    idx = torch.arange(2**n)
    perm = idx.clone()                      # perm[i]: source index for output i, built by composing gates
    for c, t in ring_pairs(n):
        # gate G: out[j] = in[g(j)], g(j) = j ^ (1<<(n-1-t)) if bit_c(j) else j   (CNOT is an involution)
        g = torch.where(bit(idx, c, n) == 1, idx ^ (1 << (n - 1 - t)), idx)
        perm = perm[g]                      # new_out[j] = prev_out[g(j)] = in[perm[g(j)]]
    return perm


def cz_ring_signs(n: int) -> torch.Tensor:
    idx = torch.arange(2**n)
    s = torch.ones(2**n, dtype=torch.float64)
    for c, t in ring_pairs(n):
        s = s * (1 - 2.0 * (bit(idx, c, n) & bit(idx, t, n)).to(torch.float64))
    return s


def z_signs(n: int) -> torch.Tensor:
    """[2^n, n] matrix of (+1/-1) eigenvalues of Z_q on each basis state."""
    idx = torch.arange(2**n)
    return torch.stack([1 - 2.0 * bit(idx, q, n).to(torch.float64) for q in range(n)], dim=1)


def apply_1q(psi: torch.Tensor, gate: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """psi[B,2^n], gate[2,2] -> gate acting on qubit q."""
    B = psi.shape[0]
    v = psi.reshape(B * 2**q, 2, 2 ** (n - q - 1))
    return torch.einsum("ab,xbz->xaz", gate, v).reshape(B, -1)


def apply_1q_batched(psi: torch.Tensor, gate: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """psi[B,2^n], gate[B,2,2] (a different gate per sample) acting on qubit q."""
    B = psi.shape[0]
    v = psi.reshape(B, 2**q, 2, 2 ** (n - q - 1))
    return torch.einsum("bij,bpjr->bpir", gate, v).reshape(B, -1)


def ry_b(theta: torch.Tensor) -> torch.Tensor:
    """theta[B] -> [B,2,2] RY gates."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(CDTYPE)


def rz_b(phi: torch.Tensor) -> torch.Tensor:
    e = torch.exp(-0.5j * phi.to(CDTYPE))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)


def ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(CDTYPE)


def rz(phi: torch.Tensor) -> torch.Tensor:
    e = torch.exp(-0.5j * phi.to(CDTYPE))
    z = torch.zeros_like(e)
    return torch.stack([torch.stack([e, z]), torch.stack([z, e.conj()])])


class TorchVQC(nn.Module):
    """Variational circuit: encode -> L x [rotations on every qubit ; entangling ring] -> <Z_i>, i < K.

    Parameters: theta[L, n, n_rot] ~ N(0, init_std^2); optional per-output ``scale`` and ``proj`` (classical, counted).
    """

    def __init__(
        self, n_qubits: int = 8, n_layers: int = 4, n_outputs: int = 8, rotations: str = "ry", entangler: str = "cnot",
        encoding: str = "amplitude", init_std: float = 0.1, trainable_scale: bool = False, input_projection: bool = False,
        input_dim: int = 256, equator_init: bool = False, readout_affine: bool = False, feat_dim: int = 20, reupload_std: float = 0.2,
    ) -> None:
        super().__init__()
        assert rotations in ("ry", "ryrz") and entangler in ("cnot", "cz", "none") and encoding in ("amplitude", "angle", "reupload")
        assert n_outputs <= n_qubits
        self.n, self.L, self.K = n_qubits, n_layers, n_outputs
        self.rotations, self.entangler, self.encoding = rotations, entangler, encoding
        n_rot = {"ry": 1, "ryrz": 2}[rotations]
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, n_rot, dtype=torch.float64) * init_std)
        self.scale = nn.Parameter(torch.ones(n_outputs, dtype=torch.float64)) if trainable_scale else None
        self.equator_init, self.feat_dim = equator_init, feat_dim
        # exploratory designs (DECISIONS D22): trainable per-output affine readout; data re-uploading from the compact 20-d feature block
        self.r_scale = nn.Parameter(torch.ones(n_outputs, dtype=torch.float64)) if readout_affine else None
        self.r_bias = nn.Parameter(torch.zeros(n_outputs, dtype=torch.float64)) if readout_affine else None
        self.W: nn.Linear | None = None
        if encoding == "reupload":
            self.W = nn.Linear(feat_dim, n_layers * n_qubits * n_rot, bias=False, dtype=torch.float64)
            with torch.no_grad():
                self.W.weight.normal_(0.0, reupload_std)
        self.proj: nn.Linear | None = None
        out_dim = 2**n_qubits if encoding == "amplitude" else n_qubits
        if input_projection:
            self.proj = nn.Linear(input_dim, out_dim, dtype=torch.float64)
            with torch.no_grad():  # identity-like start (exact identity when input_dim == out_dim)
                self.proj.weight.zero_()
                m = min(input_dim, out_dim)
                self.proj.weight[:m, :m] = torch.eye(m, dtype=torch.float64)
                self.proj.bias.zero_()
        elif encoding == "angle":
            self.proj = None  # angle encoding without projection uses x[:, :n] directly
        self.register_buffer("_perm", cnot_ring_permutation(n_qubits) if entangler == "cnot" else torch.arange(2**n_qubits), persistent=False)
        self.register_buffer("_cz", cz_ring_signs(n_qubits) if entangler == "cz" else torch.ones(2**n_qubits, dtype=torch.float64), persistent=False)
        self.register_buffer("_zs", z_signs(n_qubits), persistent=False)

    # ------------------------------------------------------------------ pieces
    def _init_state(self, B: int) -> torch.Tensor:
        """|0...0>, or the 'equator' product state RY(pi/2)^n |0...0> (all <Z_i> = 0, maximal sensitivity)."""
        N = 2**self.n
        if self.equator_init:
            return torch.full((B, N), 1.0 / math.sqrt(N), dtype=torch.float64).to(CDTYPE)
        psi = torch.zeros(B, N, dtype=torch.float64)
        psi[:, 0] = 1.0
        return psi.to(CDTYPE)

    def reupload_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample extra rotation angles [B, L, n, n_rot] = W x[:, :feat_dim] (added to theta in every layer)."""
        xc = fit_to_dim(x.to(torch.float64), self.feat_dim)
        return self.W(xc).reshape(x.shape[0], self.L, self.n, -1)

    def _ansatz_reupload(self, psi: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        for l in range(self.L):
            for q in range(self.n):
                psi = apply_1q_batched(psi, ry_b(self.theta[l, q, 0] + extra[:, l, q, 0]), q, self.n)
                if self.rotations == "ryrz":
                    psi = apply_1q_batched(psi, rz_b(self.theta[l, q, 1] + extra[:, l, q, 1]), q, self.n)
            if self.entangler == "cnot":
                psi = psi[:, self._perm]
            elif self.entangler == "cz":
                psi = psi * self._cz.to(CDTYPE)
        return psi

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(torch.float64)
        if self.encoding == "reupload":
            return self._init_state(x.shape[0]), torch.linalg.vector_norm(fit_to_dim(x, self.feat_dim), dim=-1)
        if self.proj is not None:
            x = self.proj(fit_to_dim(x, self.proj.in_features))
        if self.encoding == "amplitude":
            return amplitude_state(x, self.n)
        return angle_state(math.pi * torch.tanh(x) if self.proj is not None else x, self.n), torch.linalg.vector_norm(x, dim=-1)

    def ansatz(self, psi: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        theta = self.theta if theta is None else theta
        for l in range(self.L):
            for q in range(self.n):
                psi = apply_1q(psi, ry(theta[l, q, 0]), q, self.n)
                if self.rotations == "ryrz":
                    psi = apply_1q(psi, rz(theta[l, q, 1]), q, self.n)
            if self.entangler == "cnot":
                psi = psi[:, self._perm]
            elif self.entangler == "cz":
                psi = psi * self._cz.to(CDTYPE)
        return psi

    def final_state(self, x: torch.Tensor) -> torch.Tensor:
        psi, _ = self.encode(x)
        if self.encoding == "reupload":
            return self._ansatz_reupload(psi, self.reupload_angles(x))
        return self.ansatz(psi)

    @staticmethod
    def probs(psi: torch.Tensor) -> torch.Tensor:
        return psi.real**2 + psi.imag**2

    def expectations(self, psi: torch.Tensor) -> torch.Tensor:
        return self.probs(psi) @ self._zs[:, : self.K]

    def forward(self, x: torch.Tensor, shots: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """<Z_i> (optionally x scale). With ``shots`` the readout is sampled; gradients stay analytic (straight-through)."""
        psi = self.final_state(x)
        ez = self.expectations(psi)
        if shots is not None:
            p = self.probs(psi).detach()
            idx = torch.multinomial(p / p.sum(-1, keepdim=True), shots, replacement=True, generator=generator)  # [B, shots]
            est = self._zs[idx][..., : self.K].mean(1)
            ez = ez + (est - ez).detach()
        if self.r_scale is not None:
            ez = ez * self.r_scale + self.r_bias
        if self.scale is not None:
            ez = ez * self.scale
        return ez

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def input_norm(self, x: torch.Tensor) -> torch.Tensor:
        """||x|| (discarded by amplitude encoding) - logged as a diagnostic."""
        return self.encode(x)[1]

    @torch.no_grad()
    def entanglement_entropy(self, x: torch.Tensor, cut: int | None = None) -> torch.Tensor:
        """Von Neumann entropy (bits) of the output state across ``cut`` (default: middle bipartition)."""
        return entanglement_entropy(self.final_state(x), self.n, cut)

    def n_quantum_params(self) -> int:
        return int(self.theta.numel())

    def n_classical_params(self) -> int:
        return int(sum(p.numel() for n_, p in self.named_parameters() if n_ != "theta"))


def entanglement_entropy(psi: torch.Tensor, n: int, cut: int | None = None) -> torch.Tensor:
    cut = n // 2 if cut is None else cut
    m = psi.reshape(psi.shape[0], 2**cut, 2 ** (n - cut))
    s = torch.linalg.svdvals(m)
    p = (s**2).clamp_min(1e-30)
    return -(p * torch.log2(p)).sum(-1)
