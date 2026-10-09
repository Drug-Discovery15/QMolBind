"""PennyLane reference implementation of the same circuit (also the export path to Qiskit / hardware).

Identical conventions to ``torch_vqc.TorchVQC``: wire 0 = most significant bit, RY(theta[l,q,0]) (+ RZ(theta[l,q,1])) on
every wire, then the CNOT/CZ ring, <Z_i> for the first K wires. Amplitude encoding uses ``qml.StatePrep`` on the already
normalised vector (so the zero-vector convention matches the torch code).
"""
from __future__ import annotations

import pennylane as qml
import torch

from qumolbind.quantum.encoding import amplitude_state
from qumolbind.quantum.torch_vqc import ring_pairs


def build_qnode(
    n_qubits: int, n_layers: int, n_outputs: int, rotations: str = "ry", entangler: str = "cnot",
    device: str = "default.qubit", diff_method: str = "backprop", shots: int | None = None, interface: str = "torch",
):
    dev = qml.device(device, wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(psi, theta):
        qml.StatePrep(psi, wires=range(n_qubits), normalize=False)
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RY(theta[l, q, 0], wires=q)
                if rotations == "ryrz":
                    qml.RZ(theta[l, q, 1], wires=q)
            if entangler in ("cnot", "cz"):
                gate = qml.CNOT if entangler == "cnot" else qml.CZ
                for c, t in ring_pairs(n_qubits):
                    gate(wires=[c, t])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_outputs)]

    return circuit


class PLVQC:
    """Callable wrapper: (x[B,d], theta[L,n,n_rot]) -> <Z>[B,K] using PennyLane with parameter broadcasting."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 4, n_outputs: int = 8, rotations: str = "ry", entangler: str = "cnot",
                 device: str = "default.qubit", diff_method: str = "backprop", shots: int | None = None) -> None:
        self.n, self.L, self.K, self.rotations, self.entangler = n_qubits, n_layers, n_outputs, rotations, entangler
        self.qnode = build_qnode(n_qubits, n_layers, n_outputs, rotations, entangler, device, diff_method, shots)

    def __call__(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        psi, _ = amplitude_state(torch.as_tensor(x), self.n)
        out = self.qnode(psi, theta)  # broadcast over the leading batch axis of psi
        return torch.stack(list(out), dim=-1) if isinstance(out, (list, tuple)) else out.T

    def draw(self) -> str:
        psi = torch.zeros(2**self.n, dtype=torch.complex128)
        psi[0] = 1
        theta = torch.zeros(self.L, self.n, 2 if self.rotations == "ryrz" else 1, dtype=torch.float64)
        return qml.draw(self.qnode)(psi, theta)
