"""Qiskit export + hardware-cost report (two-qubit gate count / depth after transpiling to a Heron-like device).

Qubit-order bridge: our wire 0 is the most-significant bit; Qiskit is little-endian, so wire w <-> qiskit qubit n-1-w
(StatePreparation(psi) then lands psi's index bits on the right qubits with no reordering of psi).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

from qumolbind.quantum.torch_vqc import ring_pairs

HERON_BASIS = ["cz", "rz", "sx", "x"]  # IBM Heron-class native set (CZ, not ECR)


def build_circuit(theta: np.ndarray, psi: np.ndarray | None, n: int, rotations: str = "ry", entangler: str = "cnot",
                  equator_init: bool = False) -> QuantumCircuit:
    """Encoding (StatePreparation of ``psi``; skipped if None) + ansatz, in the torch/PennyLane convention.

    ``equator_init`` prepares RY(pi/2)^n |0..0> first (re-uploading designs); there ``theta`` holds the per-sample total angles."""
    qc = QuantumCircuit(n)
    if equator_init:
        for q in range(n):
            qc.ry(np.pi / 2, q)
    if psi is not None:
        qc.append(StatePreparation(np.asarray(psi, dtype=complex)), list(range(n)))
    qb = lambda w: n - 1 - w
    L = theta.shape[0]
    for l in range(L):
        for w in range(n):
            qc.ry(float(theta[l, w, 0]), qb(w))
            if rotations == "ryrz":
                qc.rz(float(theta[l, w, 1]), qb(w))
        if entangler in ("cnot", "cz"):
            for c, t in ring_pairs(n):
                (qc.cx if entangler == "cnot" else qc.cz)(qb(c), qb(t))
    return qc


def z_observables(n: int, K: int) -> list[SparsePauliOp]:
    """Z on wire w (< K) as a Qiskit operator (qubit n-1-w)."""
    obs = []
    for w in range(K):
        s = ["I"] * n
        s[w] = "Z"  # Pauli strings are written highest qubit first, so index w == qubit n-1-w
        obs.append(SparsePauliOp("".join(s)))
    return obs


def exact_expectations(qc: QuantumCircuit, K: int) -> np.ndarray:
    sv = Statevector(qc)
    return np.array([float(sv.expectation_value(o).real) for o in z_observables(qc.num_qubits, K)])


def heron_like_coupling(n: int) -> CouplingMap:
    """Smallest heavy-hex lattice (distance 3 = 19 qubits, distance 5 = 65) with >= n qubits."""
    for d in (3, 5, 7):
        cm = CouplingMap.from_heavy_hex(d)
        if cm.size() >= n:
            return cm
    raise ValueError(n)


def transpile_cost(qc: QuantumCircuit, n: int, opt_level: int = 3, seed: int = 0) -> dict:
    tq = transpile(qc, basis_gates=HERON_BASIS, coupling_map=heron_like_coupling(n), optimization_level=opt_level, seed_transpiler=seed)
    ops = tq.count_ops()
    return {"two_qubit_gates": int(ops.get("cz", 0)), "depth": int(tq.depth()), "total_gates": int(sum(ops.values()))}


def hardware_cost_report(ns=(4, 6, 8, 10), n_layers: int = 4, rotations: str = "ry", entangler: str = "cnot", n_states: int = 5,
                         seed: int = 0, out_csv: str | Path | None = None) -> list[dict]:
    """Two-qubit gate count / depth of [amplitude state prep + ansatz] vs [ansatz alone], averaged over random states."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        theta = rng.normal(0, 0.1, (n_layers, n, 2 if rotations == "ryrz" else 1))
        full, ans = [], []
        for _ in range(n_states):
            psi = rng.normal(size=2**n)
            psi /= np.linalg.norm(psi)
            full.append(transpile_cost(build_circuit(theta, psi, n, rotations, entangler), n))
            ans.append(transpile_cost(build_circuit(theta, None, n, rotations, entangler), n))
        m = lambda rs, k: float(np.mean([r[k] for r in rs]))
        rows.append({
            "n_qubits": n, "n_layers": n_layers, "amplitude_dim": 2**n,
            "cz_encoding_plus_ansatz": m(full, "two_qubit_gates"), "depth_encoding_plus_ansatz": m(full, "depth"),
            "cz_ansatz_only": m(ans, "two_qubit_gates"), "depth_ansatz_only": m(ans, "depth"),
            "cz_state_prep_only_est": m(full, "two_qubit_gates") - m(ans, "two_qubit_gates"),
            "basis": "/".join(HERON_BASIS), "coupling": "heavy-hex", "opt_level": 3, "n_random_states": n_states,
        })
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows
