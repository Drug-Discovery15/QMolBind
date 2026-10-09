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
- Decision: CPU torch. A GPU (RTX 5070 Ti) is present and is used by OpenMM (D12), but the 8-qubit VQC and the MLPs stay on CPU as the brief advises (GPU overhead dominates at this size; not separately benchmarked here).

## D6. Target selection (programmatic screen; see docs/DATA.md for the full table)
- 1STP (streptavidin-biotin), the prompt's suggestion, is REJECTED: RCSB resolution 2.6 A > 2.5 A.
- Screen of 8 candidates (resolution <= 2.5, exactly one non-additive ligand type, 10-45 heavy atoms, 3-8 RDKit rotatable bonds, no LINK/<1.7 A contact):
  accepted 1CIL (carbonic anhydrase II / ETS, 3 rotors), 3ERT (ER-alpha / OHT, 8), 1UYD (HSP90 / PU8, 8), 1EVE, 2HYY.
- Decision: use 1CIL (smoke target, small ligand), 3ERT, 1UYD. Alternatives: 1EVE (AChE gorge, large), 2HYY (Abl/imatinib, 37 heavy atoms, DFG-out) - dropped for oracle cost.
- Caveat: 1CIL has only K=3 torsions (the prompt's lower bound), so CMA-ES in 3-D will be near-trivial there; the 8-rotor targets are the informative ones.

## D7. Data storage: manifest with sha256 instead of DVC
- Decision: HDF5 per target in data_cache/processed/ + manifest.json (sha256). Alternative: DVC (extra dependency, needs remote). Prompt allows either.
- BindingDB: censored values (<, >) dropped; dedupe by canonical SMILES preferring Kd > Ki > IC50, median pAffinity within the preferred type; `measurement` and `n_measurements` kept.

## D8. Protein prep
- pdbfixer (heavy atoms, no invented loops; pH 7.4 hydrogens via Modeller). Chains kept = those with a heavy atom within 6 A of the ligand.
- Catalytic/structural metals ZN, MG, CA, MN within the pocket are KEPT (prompt says remove heteroatoms except the ligand): 1CIL's sulfonamide binds the active-site Zn, and without it the target is meaningless. They are modelled as non-bonded ions (amber14 ion templates).
- Pocket truncation to residues within 12 A of the ligand is applied by default for ALL targets (not only if slow), with ACE/NME caps built from the real coordinates of the removed neighbours and gaps <= 3 residues filled. Reason: keeps oracle cost low; protein is frozen anyway. 1CIL: 142 res / 2009 atoms; 3ERT: 132 / 2051; 1UYD: 147 / 2150.

## D9. Torsion application: own Rodrigues rotation, not RDKit SetDihedral
- Alternative: rdMolTransforms.SetDihedralDeg (mutates a Conformer, ~10x slower in Python, ambiguous about which side moves for ring-adjacent bonds). Mine rotates the side not containing the root fragment about the current bond axis; tests verify bond lengths preserved, exact dihedral change, +theta/-theta restore (< 1e-6 A) and agreement with RDKit's GetDihedralDeg.
- Rotatable bonds: RDKit *strict* SMARTS (no amides, no terminal CF3/tBu); K cap keeps the torsions that move the most atoms. Root fragment = largest heavy-atom component after cutting rotatable bonds.

## D10. RMSD symmetry
- Automorphisms are computed on the topology-only heavy-atom graph (bond orders/charges/aromaticity ignored) so carboxylate, sulfonyl and nitro oxygens are interchangeable. RDKit's CalcRMS uses typed matching and would count a carboxylate O swap as an error; tests compare against CalcRMS on a molecule where the two coincide. RMSD is in-place (no superposition), evaluation only.

## D11. Ligand force field (replaces openff-2.1.0 / GAFF2, which need openff-toolkit / AmberTools, absent from PyPI)
- Decision: generated in `sim/params.py`: MMFF94 partial charges (Gasteiger fallback, logged), UFF Lennard-Jones (polar H eps=0), harmonic bonds/angles frozen at the native geometry (pose-independent, cancel in E_int), AMBER-style 1-4 scaling (0.8333/0.5). Protein: amber14 (ff14SB). Solvent: GBn2 via OpenMM's `implicit/gbn2.xml`.
- Alternatives: project-local micromamba env with openff-toolkit (AmberTools is not available for win-64 anyway; AM1-BCC impossible); Gasteiger-only.
- Consequences: absolute energies are NOT comparable to Sage/GAFF2 results; relative pose ranking is what the study needs. Neutral sulfonamide at the 1CIL zinc site gives a positive native E_int (sulfonamide N is deprotonated in reality) - an acknowledged limitation; the native pose still ranks below random-torsion poses.

## D12. Oracle platform: OpenCL (GPU) by default, CPU fallback
- Measured (see `results/oracle_benchmark.csv`, produced by `scripts/benchmark_oracle.py`, and the table in REPORT.md / docs/ARCHITECTURE.md): the CPU platform with 1 thread needs ~0.5 s per GBn2 evaluation, OpenCL on the RTX 5070 Ti is two orders of magnitude faster end-to-end. The pip OpenMM build has no fast CPU CustomGBForce path here.
- Decision: `oracle_platform: auto` -> OpenCL when available else CPU. This deviates from "one OpenMM context per worker, single-threaded CPU" in spirit (each worker still has its own context). The smoke target "<15 min on CPU" therefore assumes a GPU for OpenMM; without one, smoke would take hours (documented in README).
- Mixed precision differs from double by ~0.02 kJ/mol on solvation terms; tests that need tight tolerances use `precision=double`.

## D13. Objective = E_int + MMFF strain; reward uses a signed-log transform
- E_int = E_complex - E_protein - E_ligand cancels all ligand-internal terms, so it cannot penalise intramolecular clashes in torsion space. Score = E_int + (MMFF94 intramolecular energy of the pose - that of the native pose); `include_strain` can disable it.
- Clash energies reach 1e9 kJ/mol; reward = -(e_t - e_{t-1}) with e = symlog(score/100 kJ/mol) (monotone, so optima are unchanged), clipped to +-10; the clip rate is logged. `energy_transform: linear` gives the prompt's literal -(dE)/scale.
- State: fixed layout with K_max=8 torsion slots (sin 8, cos 8) so the layout is target independent (needed for transfer E8).
- Reset evaluates the oracle once (counted in the budget).

## D14. Baseline protocol
- All methods run through one `Tracker` that hard-caps TOTAL oracle calls at B (env resets count) and records best-energy-so-far; success = RMSD of the best-energy pose < 2 A.
- PPO: CleanRL-style, own implementation; reward normalised by running std of discounted returns; time-limit truncations bootstrapped; n_envs=4 x 20 steps per update.
- Equal tuning effort = 3 trials per method on separate tuning seeds (`--tune`): PPO methods sweep actor lr {3e-3,1e-2,3e-2}; hill-climb step {10,20,40} deg; CMA-ES sigma0 {10,20,40} deg; random search has no hyper-parameter (3 trial seeds). Smoke runs skip tuning and use the middle value.
- CMA-ES (sigma0=40) was initially worse than random search on 3ERT: the clash landscape is rugged at the 40 deg scale (energies up to 1e11), step size diverged. Implementation validated on a smooth periodic objective (tests/baselines); grid was moved to smaller sigma. This is a property of the landscape, not tuned away.
- `ppo_mlp_matched`: exact matching to a 32-parameter VQC is infeasible with the raw 256-d input (a width-1 hidden layer already has 263 parameters); the closest feasible network is used and both counts are reported. Against VQC variant (ii) (trainable 256->256 projection, 65,792 params) matching is exact to <1%.
- 1CIL has only 3 small torsions with a fixed root: every random start already has RMSD ~1.3 A, so "success" is uninformative there. Reported alongside the random-start success rate; 3ERT and 1UYD (8 torsions) are the informative targets.

## D15. Encoders (Stage 5)
- `pyg-lib` / `torch-cluster` / `torch-sparse` have no wheels for this torch build, so PyG's DimeNet++/SchNet cannot call `radius_graph` / `triplets`. Decision: pure-torch replacements (`encoders/gnn.py`, monkey-patched into PyG), verified against brute force; DimeNet++ stays the default. Alternatives: fall back to SchNet/GINE (both also implemented and tested).
- The affinity head uses SiLU, not ReLU: PyG zero-initialises DimeNet's output layer, the 64-d embedding starts at exactly 0, ReLU(0) has zero gradient and the model stayed at the mean prediction (observed: train MSE 1.000 for 11 epochs).
- Training set capped at 2500 molecules per target (random subsample, seed 0) to bound conformer generation time; scaffold split 80/10/10. Report whatever comes out: see the affinity table (the LightGBM-ECFP4 baseline beat the GNN on the targets trained so far).
- BindingDB returned HTTP 503 for AChE (P04058) during the build: 1EVE has structure data but no affinity set, so no GNN/ligand embedding for it (embeddings are off by default). A first version of the fetcher silently substituted another target's fixture; fixed (fixture only for its own UniProt, retries, explicit failure) and the bogus files deleted.
- ESM-2 pocket embedding: mean-pooled last hidden state, fixed random projection 320 -> 32 (JL-style), L2-normalised.

## D16. Targets: 1CIL demoted to a test fixture (supersedes the target choice of D6)
- Zinc-site carbonic anhydrase with a bare non-bonded Zn2+ ion: the native pose has a positive interaction energy (neutral sulfonamide next to Zn2+) and MD collapses within ~0.4 ps (ligand N falls onto Zn2+; deep Coulomb well, shallow LJ). Real treatments need bonded/cationic-dummy Zn models or a deprotonated sulfonamide.
- Decision: main targets = 3ERT, 1UYD, 1EVE (metal-free, all 6-8 torsions). 1CIL remains in the data screen and as a small committed fixture for unit tests (K=3, fast). Smoke target = 1EVE (strained random starts still relax in Level-2 MD); offline fallback = committed 3ERT fixture.

## D17. Level-2 oracle details
- Ligand torsions for MD are heuristic generic periodic terms (barriers: double bond 100, amide 60, sp2-sp2 20, sp3-sp3 11, sp2-sp3 2 kJ/mol) - fast-oracle poses never use them.
- OpenCL force accumulation is fixed-point: forces saturate at 2^31 kJ/mol/nm for strained poses, so the minimiser stalls and MD diverges (NaN). Protocol: (1) torsion-space pre-relaxation against the Level-1 score (150 evals, counted as Level-2 cost, not in the L1 budget), reject if still > 1.5e3 kJ/mol; (2) staged Cartesian minimisation; (3) 30 K -> 300 K heating at 0.5 fs for ~1 ps; (4) 2 fs production with H-bond constraints. Failures are recorded (`l2_failed`) and excluded from surrogate training.
- MD length: smoke 2 ps (plumbing only), full 100 ps; first half of frames discarded. Protein frozen (mass 0); single trajectory; no entropy.
- Consequence observed in the smoke run: strained candidates relax into similar basins, so L2 dG is only weakly pose-dependent there (Spearman(L1, L2) is reported, not assumed).

## D18. Quantum policy details
- Readout of the first K qubits requires K <= n; the 8-torsion targets therefore need n >= 8.
- Amplitude encoding of a vector whose length differs from 2^n pads with zeros / truncates (n=4 keeps only the sin/cos torsion block of the state); zero input maps to |0...0>.
- Angle encoding needs one feature per qubit: it uses a trainable d -> n projection (counted) with RY(pi tanh(.)) angles (variant `ppo_vqc_angle`).
- Shot noise (E5) is simulated in the forward pass (sampled readout) with the analytic gradient (straight-through), i.e. NOT a hardware-style parameter-shift estimator. Gate noise (E6) is evaluation-only (density matrix, depolarizing on both qubits of every CNOT/CZ, matches PennyLane `default.mixed` to 1e-8) - policies are not trained under noise.
- Near-identity init `theta ~ N(0, 0.1^2)` + amplitude encoding gives initial means far from 0 for qubits whose basis-state bits are mostly 0 (the state occupies low basis indices); this is a property of the specified design, not tuned away.
- `ppo_mlp_matched`: see D14; for the 32-parameter VQC no MLP on the raw 256-d input can be matched.

## D19. Experiments E4-E8 (what was and was not done)
- E4 reports gradient variance at random init (>= 200 inits, near-identity and uniform inits, cost = local <Z_0> on real env states); no trained n/L scaling sweep (K <= n and runtime).
- E7 uses Qiskit-Aer matrix_product_state with `initialize` (exact at chi = 2^(n/2) = 16, tested); the pre-registered clause uses chi = 4 (docs/PREREGISTRATION.md).
- E8 transfers 3ERT -> 1UYD (both K=8) because the MLP output layer is K-specific; fine-tune budget = 25% of B; same seeds shifted by 500 for the destination.
- The MPI-style "vectorised rollouts via multiprocessing" exists (`env/vec.py::SubprocVecEnv`, tested) but experiments use the in-process `SyncVecEnv` sharing one oracle: with the GPU oracle the OpenCL context, not Python, is the bottleneck and a shared context avoids one GPU context per worker.

## D20. Packaging
- The Docker CLI is installed on the build machine but the Docker Desktop daemon was not running; starting a GUI daemon was not done without being asked, so the Dockerfile has NOT been built or verified (it follows environment.yml: conda-forge Python 3.11 + pip for the rest). The tested environment is the Windows pip venv. In a container without a GPU the oracle falls back to the CPU platform (D12).
- `requirements-lock.txt` is a `pip freeze` of the tested `.venv` (Windows, Python 3.12, CPU torch).
- Smoke runtime measured on the build machine (RTX 5070 Ti, 24 cores): see `results/smoke_run.log` ("real" line).

## D21. "pilot" experiment (what the generated REPORT can and cannot say)
- The brief asks for >= 10 seeds and B = 20,000. At ~150-250 oracle evals/s with 15 methods x 3 targets that is days of wall-clock on one machine, so the executed configs are `smoke` (B=500, 2 seeds; pipeline proof only) and `pilot` (B=2000, 6 seeds, 3 targets, 3-trial tuning for the E1 methods). `full` (B=20000, 10 seeds) is defined and runnable (`EXP=full`) but was not executed here.
- With 6 seeds the bootstrap CIs are wide and the pre-registered criterion has little power; a NO DEMONSTRATED ADVANTAGE verdict at pilot scale means "not shown at this scale", not "shown to be absent".
- Tuning: E1 methods get 3 trials (1 tuning seed) per target; ablation variants (E2/E3/E5) reuse the lr tuned for `ppo_vqc` on that target so they differ from it in one factor only.
