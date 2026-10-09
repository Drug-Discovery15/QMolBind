# QuMolBind

Hybrid quantum-classical reinforcement learning for **ligand torsion search in a frozen protein pocket**. A policy proposes torsion changes; an
OpenMM-based oracle scores the pose; the policy is trained with PPO. The policy's mean network is either a classical MLP or a variational
quantum circuit (VQC, amplitude-encoded 8-qubit state, Pauli-Z readout) - "QPPO". An active-learning layer (surrogate ensemble + UCB) decides which
poses get re-scored by a slower MD-based oracle. **The quantum circuit only proposes the next move; it never outputs a molecule.**

This repository is a research codebase. **No result is stated in this README by hand**; everything below the marker is generated from files in
`results/` by `make report`, and [REPORT.md](REPORT.md) contains every table and figure with its provenance.

<!-- RESULTS:START -->
_not yet generated - run `make report`._
<!-- RESULTS:END -->

## Quick start

```bash
python -m venv .venv && .venv/bin/activate          # Windows: .venv\Scripts\activate  (everything is installed into this venv only)
python scripts/tasks.py env        # == make env   : pip install -e .[dev,hardware] + import check
python scripts/tasks.py test       # == make test
python scripts/tasks.py smoke      # == make smoke : 1 target, B=500, 2 seeds, all methods, 2 AL rounds, report
```

`make` is optional (`Makefile` delegates to `scripts/tasks.py`; stock Windows has no make). Targets: `env data test smoke baselines train-q experiments report`;
`EXP=smoke|full` selects the experiment config.

**Hardware assumption.** The Level-1 oracle evaluates GBn2 implicit solvent on the OpenCL platform (a GPU). On a single CPU thread the same evaluation is
orders of magnitude slower (measured in the benchmark table of `REPORT.md` / `docs/ARCHITECTURE.md`), so `make smoke` does **not** finish in ~15 minutes on a
CPU-only machine (DECISIONS D12). The 8-qubit VQC itself runs on CPU.

## What is where

| path | content |
|---|---|
| `qumolbind/data`, `chem`, `sim`, `env` | PDB/BindingDB/ZINC loaders, target prep + torsion model + symmetry RMSD, oracles, Gymnasium env |
| `qumolbind/rl`, `baselines` | PPO (own implementation), actors/critic, random search / hill climb / CMA-ES / PPO-MLP, shared budget tracker |
| `qumolbind/quantum` | torch + PennyLane VQC (cross-checked), noise, MPS, Qiskit export, hardware-cost report, hardware-check module |
| `qumolbind/active` | surrogate ensemble, UCB, multi-fidelity records, the active-learning loop |
| `qumolbind/eval` | bootstrap/MWU statistics, metrics + the mechanical pre-registered verdict, plots, report |
| `scripts/` | one entry point per stage / experiment (E1-E8) |
| `docs/` | `DATA.md` (generated: exactly what was downloaded), `ARCHITECTURE.md`, `PREREGISTRATION.md` |
| `DECISIONS.md` | every deviation from the build brief, with alternatives and reasons |

## Scope and honest caveats

* Pilot scale: torsion-space search of a known ligand in a rigid, pocket-truncated protein; targets 3ERT, 1UYD, 1EVE (selected by a programmatic screen;
  the brief's suggested 1STP fails the resolution filter). 1CIL (zinc enzyme) is kept only as a small test fixture: a non-bonded Zn model makes its energies and MD unreliable.
* The ligand force field is generated in-repo (MMFF94 charges + UFF LJ) because OpenFF/AmberTools are not available from pip; absolute energies are not comparable to Sage/GAFF2.
* The ligand and pocket embeddings are constant vectors in a single-target run; they are off by default and exist for architecture fidelity (ablation switches provided).
* Classical baselines (hill climbing, CMA-ES, random search) are strong in 6-8 dimensions; a VQC is only reported as better under the pre-registered criterion in
  [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md), evaluated mechanically by `scripts/make_report.py`.
* Amplitude encoding of a generic vector needs ~2^n two-qubit gates (see the hardware-cost table); the optional hardware check (`scripts/run_hardware_eval.py`)
  is **off by default**, runs only inference on <= 20 states, and requires both `--allow-hardware` and `IBM_QUANTUM_TOKEN`. Its default dry run is a noise-model *simulation*, labelled as such.
* The Level-2 oracle is a short single-trajectory MD estimate without entropy; at smoke length (2 ps) it is a smoke test of the plumbing, not a binding free energy.
