"""Trainable-parameter bookkeeping for the VQC actor (used to size the matched MLP)."""
from __future__ import annotations


def vqc_param_count(
    n_qubits: int = 8, n_layers: int = 4, rotations: str = "ry", n_actions: int = 8,
    trainable_scale: bool = False, input_projection: bool = False, state_dim: int = 256,
) -> dict[str, int]:
    n_rot = {"ry": 1, "ryrz": 2}[rotations]
    quantum = n_layers * n_qubits * n_rot
    classical = (n_actions if trainable_scale else 0) + ((state_dim * 2**n_qubits + 2**n_qubits) if input_projection else 0)
    return {"quantum": quantum, "classical": classical, "total_mean_net": quantum + classical}
