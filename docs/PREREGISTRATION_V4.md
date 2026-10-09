# Pre-registration V4: greedy-acceptance learned local search (written before any V4 run)

Context: V3 (REPORT_V3.md) showed elite restarts narrow the gap but hill climbing / CMA-ES stay ahead; the VQC is at parity with classical PPO.
Hypothesis: the remaining gap is mostly *acceptance*: hill climbing keeps a move only if it lowers the energy, PPO episodes keep every move (a random walk
that must be undone). V4 gives ALL PPO policies the same two algorithmic changes and nothing else:

1. **greedy acceptance** - a proposed move is applied only if it lowers the Level-1 score; otherwise the pose is unchanged. The reward is still the
   (symlog) improvement of the *proposal* (negative when worse), so the policy learns which moves help.
2. **elite restarts** with p = 0.9 (V3) and a smaller initial step size (init log-std swept, see below).

With a zero mean this is an adaptive-step-size random local search, i.e. essentially hill climbing. A **diagnostic classical policy `ppo_zero_mean`**
(mean identically 0, only the step size is learned) is therefore included as a baseline: it measures what a mean network - classical or quantum -
adds on top of the local-search mechanics. If the VQC only matches `ppo_zero_mean` / hill climbing, its parity means "the quantum part adds nothing here".

## Phase A - exploration (hypothesis generation; no claims)

* PPO designs: `ppo_zero_mean`, `ppo_mlp_matched`, `ppo_mlp_large`, `ppo_mlp_matched_reup`, `ppo_vqc`, `ppo_vqc_aff`, `ppo_vqc_reup` (PPO setting A: 4 envs, entropy 0).
* Equal effort: 6 trials per method = lr {3e-3, 1e-2, 3e-2} x initial log-std {-1.6, -0.9}. Seeds {500, 501}, targets 3ert and 1uyd, B = 2000.
* Score = mean over the 4 cells of symlog(best_energy / 100 kJ/mol). Subject = best-scoring VQC among `ppo_vqc`, `ppo_vqc_aff`, `ppo_vqc_reup`.

## Phase B - confirmation (the only place a claim can be made)

* Fresh seeds {600..609}; 3ert, 1uyd, 1eve; B = 2000. Baselines: random_search, hill_climb, cmaes (pilot 3-trial tuning), `ppo_zero_mean`,
  `ppo_mlp_matched`, `ppo_mlp_large`, `ppo_mlp_matched_reup` (each with its Phase-A configuration and the V4 mechanics).
* **Advantage verdict** (unchanged rule): the subject beats the best baseline on >= 2 targets (95% bootstrap CI of the median-energy difference or of the
  success-rate difference excludes 0 in its favour) AND Aer-MPS chi=4 error on its visited states > 0.05.
* **Parity verdict ("same level as the best classical baseline")**: on EVERY target, the lower end of the 95% bootstrap CI of
  (best-baseline median best energy - subject median best energy) is >= -100 kJ/mol (non-inferiority margin of 100 kJ/mol, fixed now; roughly the
  across-seed spread of hill climbing at B=2000), and the subject's success rate is not significantly lower (upper CI end of the success difference
  baseline - subject is < 0.3). Reported as `PARITY` / `NOT PARITY` together with the advantage verdict.
* Diagnostic (reported, not a criterion): subject vs `ppo_zero_mean` (does the mean network add anything?).

## Honest limits stated in advance

* A parity result produced by a policy that merely matches `ppo_zero_mean` would be a result about classical local search, not about the VQC; the report says which.
* Every V-round uses fresh seeds; results of earlier rounds are never pooled into a claim. Several rounds were run, so any single positive round deserves a replication.
* Exact classical simulation of 8 qubits on 3 pilot targets; nothing is run on quantum hardware.
