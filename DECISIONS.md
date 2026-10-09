# DECISIONS

Each entry: decision / alternatives / reason.

## D1. Isolated venv, Python 3.12 (not conda, not 3.11)
- Decision: project-local `.venv` (user instruction: never touch the global env); Python 3.12.7 is the only interpreter present.
- Alternatives: conda-forge Python 3.11 env as in the prompt (conda not installed).
- Reason: user constraint + availability. `environment.yml` is kept for conda users, but the tested path is pip in `.venv`.

## D2. No `make` on this machine
- Decision: `Makefile` as specified, delegating to `scripts/tasks.py`; on Windows run `python scripts/tasks.py <target>`.
- Reason: stock Windows has no make.

## D3. OpenFF toolkit is not on PyPI
- Decision: `openff-toolkit`, `openff-nagl`, `openff-interchange` have no pip distributions (verified with `pip index versions`), and AmberTools (antechamber/AM1-BCC) is not pip-installable. The prompt's default `openff-2.1.0` via `SystemGenerator` is therefore unavailable in the venv. Ligand parametrization approach is decided in Stage 3 and recorded below.
- Alternatives: project-local micromamba env (would violate the venv instruction); skip.

## D4. Installed versions (first successful install)
torch 2.14.1+cpu, pennylane 0.45.1, pennylane-lightning 0.45.0, openmm 8.6.1, rdkit 2026.03.6, qiskit 2.5.2, qiskit-aer 0.17.2,
qiskit-ibm-runtime 0.50.0, torch_geometric 2.8.0.post1, transformers 5.19.0, pdbfixer 1.12.0, openmmforcefields 0.15.1, numpy 2.5.3.
Lockfile: `requirements-lock.txt` (pip freeze of `.venv`).

## D5. Torch build
- Decision: CPU torch. GPU (RTX 5070 Ti) present but 8-qubit sims and small MLPs run faster on CPU; GPU left optional.
