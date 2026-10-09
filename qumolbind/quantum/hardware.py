"""Inference-only hardware check (Stage 9). OFF by default.

* ``build_isa_pubs``: encode <= 20 states (StatePreparation) + trained ansatz, transpile to the target backend and map the
  Z_i observables through the transpiler layout (Estimator V2 pubs).
* ``run_dry``: no network. Same circuits on a *noise-model simulation* of a bundled Heron-r2 snapshot (FakeFez), labelled as
  simulation. It gives an expected error with shots, NOT a hardware measurement.
* ``run_hardware``: real device through qiskit-ibm-runtime EstimatorV2. Requires ``allow_hardware=True`` AND the
  IBM_QUANTUM_TOKEN environment variable; jobs are deleted afterwards (deletion logged). No training ever happens on hardware.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp

from qumolbind.quantum.hw_cost import build_circuit, exact_expectations, z_observables

MAX_STATES = 20


@dataclass
class HWReport:
    backend: str
    kind: str                    # 'hardware' | 'noise-model simulation (NOT hardware)'
    shots: int
    n_states: int
    mean_abs_error: float
    max_abs_error: float
    two_qubit_gates_mean: float
    depth_mean: float
    per_state_error: list[float] = field(default_factory=list)
    job_ids: list[str] = field(default_factory=list)
    deleted_jobs: list[str] = field(default_factory=list)


def build_isa_pubs(theta: np.ndarray, states: np.ndarray, n: int, K: int, backend, rotations: str = "ry", entangler: str = "cnot",
                   opt_level: int = 3, seed: int = 0):
    """-> (pubs, exact[n_states,K], cost dict). ``states`` are the (already normalised) amplitude vectors, 2^n each."""
    if len(states) > MAX_STATES:
        raise ValueError(f"at most {MAX_STATES} states may be sent to hardware (got {len(states)})")
    obs = z_observables(n, K)
    pubs, exact, cz, depth = [], [], [], []
    for psi in states:
        qc = build_circuit(theta, psi, n, rotations, entangler)
        isa = transpile(qc, backend=backend, optimization_level=opt_level, seed_transpiler=seed)
        ops = isa.count_ops()
        cz.append(int(ops.get("cz", 0) + ops.get("ecr", 0) + ops.get("cx", 0)))
        depth.append(isa.depth())
        pubs.append((isa, [o.apply_layout(isa.layout) for o in obs]))
        exact.append(exact_expectations(qc, K))
    return pubs, np.stack(exact), {"two_qubit_gates_mean": float(np.mean(cz)), "depth_mean": float(np.mean(depth))}


def _summarise(est: np.ndarray, exact: np.ndarray, backend: str, kind: str, shots: int, cost: dict, **kw) -> HWReport:
    err = np.abs(est - exact)
    return HWReport(backend, kind, shots, len(exact), float(err.mean()), float(err.max()), cost["two_qubit_gates_mean"], cost["depth_mean"],
                    per_state_error=[float(e) for e in err.mean(1)], **kw)


def run_dry(theta: np.ndarray, states: np.ndarray, n: int, K: int, shots: int = 4096, rotations: str = "ry", entangler: str = "cnot") -> HWReport:
    """Offline: transpile for FakeFez and estimate with Aer using that backend's noise model."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    from qiskit_ibm_runtime.fake_provider import FakeFez

    fake = FakeFez()
    pubs, exact, cost = build_isa_pubs(theta, states, n, K, fake, rotations, entangler)
    sim = AerSimulator.from_backend(fake)
    est = AerEstimator.from_backend(sim, options={"default_precision": 1.0 / np.sqrt(shots)})
    res = est.run(pubs).result()
    vals = np.stack([np.asarray(r.data.evs, dtype=float) for r in res])
    return _summarise(vals, exact, f"{fake.name} (offline noise-model simulation)", "noise-model simulation (NOT hardware)", shots, cost)


def run_hardware(theta: np.ndarray, states: np.ndarray, n: int, K: int, shots: int = 4096, backend_name: str | None = None,
                 allow_hardware: bool = False, rotations: str = "ry", entangler: str = "cnot") -> HWReport:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not allow_hardware or not token:
        raise PermissionError("hardware disabled: pass --allow-hardware AND set IBM_QUANTUM_TOKEN")
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=os.environ.get("IBM_QUANTUM_INSTANCE"))
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False, min_num_qubits=max(n, 5))
    pubs, exact, cost = build_isa_pubs(theta, states, n, K, backend, rotations, entangler)
    est = EstimatorV2(mode=backend)
    est.options.default_shots = shots
    job = est.run(pubs)
    job_ids = [job.job_id()]
    res = job.result()
    vals = np.stack([np.asarray(r.data.evs, dtype=float) for r in res])
    deleted = []
    for jid in job_ids:  # delete job data from the platform when the API allows it, and log the outcome
        try:
            service.delete_job(jid)
            deleted.append(jid)
        except Exception as e:  # noqa: BLE001
            print(f"[hardware] could not delete job {jid}: {e}")
    print(f"[hardware] jobs {job_ids}; deleted: {deleted}")
    return _summarise(vals, exact, backend.name, "hardware", shots, cost, job_ids=job_ids, deleted_jobs=deleted)
