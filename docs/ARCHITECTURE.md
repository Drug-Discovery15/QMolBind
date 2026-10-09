# QuMolBind architecture

## What is being learned

A ligand sits in a **frozen** protein pocket (pocket-truncated to residues within 12 A of the crystal ligand, ACE/NME capped).
A policy repeatedly proposes torsion changes `delta in [-1,1]^K * 60 deg`; an OpenMM-based **oracle** scores the new pose; the policy is
trained with PPO. The actor (the *mean network* only) is either a classical MLP or a variational quantum circuit (VQC).
**Quantum = mean network only.** The Gaussian `log_std`, the critic, PPO, the environment and the oracle are classical in every method.
The circuit proposes the next move; it never outputs a molecule.

## Data flow

```mermaid
flowchart LR
    subgraph data[Data / preparation]
        PDB[RCSB PDB entry] --> PREP[prep.py: pdbfixer, pocket truncation + caps]
        CCD[CCD SMILES] --> LIG[ligand.py: bond orders, torsions, root fragment]
        BDB[BindingDB per-target slice] --> AFF[affinity_train.py: GNN vs LightGBM-ECFP4]
    end
    PREP --> ORACLE
    LIG --> ENV
    subgraph oracle[Oracles / fidelity levels]
        ORACLE[Level 1 fast oracle: E_int = vdW + Coulomb + dGBn2, + MMFF strain]
        SLOW[Level 2 slow oracle: torsion pre-relax, Langevin MD, MM-GBSA-style dG]
        SUR[Level 0 surrogate: 5 MLPs, mean +- std]
    end
    ENV[PoseEnv: state d=256, reward = -d symlog energy] -->|pose| ORACLE
    ORACLE -->|energy terms| ENV
    ENV -->|state| ACTOR{{Actor mean network}}
    ACTOR -->|MLP or VQC| ENV
    subgraph vqc[VQC actor]
        ENC[amplitude encoding x/||x||, 8 qubits] --> ANS[L=4 x RY + CNOT ring] --> RO[Z_i, i<K]
    end
    ACTOR -.-> ENC
    ENV --> PPO[PPO: GAE, clip, classical critic]
    PPO --> ACTOR
    ENV -->|visited poses| CAND[M candidates per round]
    CAND --> SUR
    SUR -->|UCB: -mu + beta sigma| SEL[top-b]
    SEL --> SLOW
    SLOW -->|dG labels| SUR
    SLOW -.->|refit critic| PPO
    AFF -.->|optional frozen 64-d embedding| ENV
```

## State (`env/state.py`)

`[sin(8 torsion slots) | cos(8) | symlog-normalised vdW, elec, solv (3) | t/T | ligand GNN emb (64, optional) | pocket ESM emb (32, optional)]`,
zero-padded to **d = 256**; layout fixed across targets. RMSD and native coordinates are never in the state or the reward (a unit test
corrupts the RMSD reference and checks the state is unchanged). Both embeddings are constant vectors for a single-target run; they are
off by default (`use_ligand_emb`, `use_pocket_emb` = false) and are an architecture-fidelity feature with an explicit ablation switch.

## Oracle (`sim/`)

* **Force field**: amber14 protein; ligand parametrised in `sim/params.py` (MMFF94 charges, UFF Lennard-Jones, frozen-geometry bonded terms) because
  OpenFF/AmberTools are not pip-installable (DECISIONS D11). Solvent: GBn2 (OpenMM `implicit/gbn2.xml`).
* **Level 1** `E_int = E_complex - E_protein - E_ligand = vdW + Coulomb + solvation`. vdW/Coulomb between ligand and protein are evaluated
  analytically with the exact parameters OpenMM holds; the solvation term is the GBn2 energy difference from OpenMM. A test checks the sum against
  the full OpenMM energies (`reference_interaction`). The optimised score adds an MMFF94 strain term (D13). Every call is counted.
* **Level 2** short Langevin MD (ligand flexible, protein frozen) after a torsion-space pre-relaxation; `dG = <E_complex - E_protein - E_ligand>`
  over production frames; no entropy; terms listed in `oracle_slow.py`.
* **Platform**: OpenCL (GPU) by default - see the benchmark below; CPU is a (very slow) fallback.

## Quantum core (`quantum/`)

| piece | file | note |
|---|---|---|
| amplitude / angle encoding | `encoding.py` | amplitude: `x/(\|\|x\|\|+1e-8)`, norm discarded (logged); zero vector -> `\|0..0>` |
| torch simulator | `torch_vqc.py` | complex128 batched statevector, reshape-based gates, exact autograd, straight-through shot sampling |
| PennyLane reference | `pl_vqc.py` | `default.qubit`, parameter broadcasting, backprop / parameter-shift |
| gate noise | `noise.py` | density matrix, depolarizing p after every CNOT/CZ (matches `default.mixed`) |
| MPS | `mps.py` | Qiskit-Aer matrix_product_state, bond dimension chi |
| hardware cost / export | `hw_cost.py`, `hardware.py` | StatePreparation + ansatz -> Heron basis (cz/rz/sx/x), heavy-hex |

Ansatz: `theta ~ N(0, 0.1^2)`, per layer RY (optionally RZ) on every qubit then a CNOT (or CZ) ring. Parameter count = `L x n x #rot`.

## Active learning (`active/`)

Per round: PPO continues on Level 1 -> M diverse low-energy poses visited by the policy -> Level-0 surrogate predicts (mu, sigma) of the Level-2 value
(before any label exists) -> UCB picks top-b -> Level-2 MD -> surrogate refit on all L2 labels, critic refit on accumulated returns -> repeat.
Round 0 has no labels and uses the lowest-L1 candidates (warm start). Every record is tagged with its fidelity.

## Fairness (`baselines/`)

One `Tracker` caps TOTAL oracle calls (resets included) for every method; same env, state, seeds; 3 tuning trials per method; matched and large MLPs
plus two VQC input variants. CMA-ES, hill climbing and random search act on the same objective through the same oracle.

## Oracle benchmark (generated)

<!-- BENCH:START -->
Single process, full fast-oracle evaluation per pose (pair terms + GBn2 complex + GBn2 ligand + MMFF strain), measured on this machine:

| target | n_atoms | platform | precision | threads | device | evals_per_s | n_evals |
|---|---|---|---|---|---|---|---|
| 3ert | 2109 | OpenCL | mixed | nan | NVIDIA GeForce RTX 5070 Ti | 241.0 | 100 |
| 3ert | 2109 | OpenCL | double | nan | NVIDIA GeForce RTX 5070 Ti | 91.1 | 100 |
| 3ert | 2109 | CPU | nan | 1 | nan | 1.9 | 6 |
| 3ert | 2109 | CPU | nan | 8 | nan | 9.2 | 6 |
| 1uyd | 2202 | OpenCL | mixed | nan | NVIDIA GeForce RTX 5070 Ti | 250.1 | 100 |
| 1uyd | 2202 | OpenCL | double | nan | NVIDIA GeForce RTX 5070 Ti | 88.2 | 100 |
| 1uyd | 2202 | CPU | nan | 1 | nan | 1.7 | 6 |
| 1uyd | 2202 | CPU | nan | 8 | nan | 4.7 | 6 |
| 1eve | 2659 | OpenCL | mixed | nan | NVIDIA GeForce RTX 5070 Ti | 154.3 | 100 |
| 1eve | 2659 | OpenCL | double | nan | NVIDIA GeForce RTX 5070 Ti | 63.0 | 100 |
| 1eve | 2659 | CPU | nan | 1 | nan | 1.2 | 6 |
| 1eve | 2659 | CPU | nan | 8 | nan | 5.9 | 6 |
<!-- BENCH:END -->
