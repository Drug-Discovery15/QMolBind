"""Classical MPS simulation of the VQC (Qiskit-Aer matrix_product_state, bond dimension chi) for the simulability test (E7).

The input state is loaded with ``initialize`` (exact MPS of the amplitude vector, then truncated to chi by Aer), the
ansatz is applied gate by gate with truncation to chi after each two-qubit gate (long-range ring-closing gate handled by
Aer's internal swaps). chi = 2^(n/2) is exact for n qubits.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qumolbind.quantum.hw_cost import build_circuit, z_observables


def mps_expectations(theta: np.ndarray, psi: np.ndarray, n: int, K: int, chi: int, rotations: str = "ry", entangler: str = "cnot") -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.initialize(np.asarray(psi, dtype=complex), list(range(n)))
    qc.compose(build_circuit(theta, None, n, rotations, entangler), inplace=True)
    qc.save_statevector()
    sim = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=int(chi))
    sv = Statevector(np.asarray(sim.run(qc).result().get_statevector(qc)))
    return np.array([float(sv.expectation_value(o).real) for o in z_observables(n, K)])
