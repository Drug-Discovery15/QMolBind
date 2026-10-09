# Pre-registration V2: exploratory VQC redesign + confirmation (written before any V2 run)

Context: the pilot (docs/PREREGISTRATION.md, REPORT_pilot) found no demonstrated advantage; `ppo_vqc` (32-parameter amplitude-encoded VQC) was
significantly worse than hill climbing on 2/3 targets and about on par with random search. This document fixes, **before running anything**, how
we look for a better quantum design without rigging the comparison.

## Phase A - exploration (hypothesis generation only; no claims)

* Designs (all new, all PPO with the same critic/PPO code): `ppo_vqc` (original), `ppo_vqc_aff` (amplitude encoding + trainable affine readout),
  `ppo_vqc_reup` (data re-uploading of the 20-d feature block, "equator" RY(pi/2) start so initial means are 0, affine readout),
  `ppo_vqc_reup_l8` (L=8), `ppo_vqc_reup_ryrz` (RY+RZ). Rationale: amplitude encoding discards the norm and the original design starts saturated.
* Equal tuning effort for every PPO method (both MLPs, a size-matched MLP for the re-uploading VQC, all VQC designs): 6 trials =
  3 learning rates {3e-3, 1e-2, 3e-2} x 2 shared PPO settings {A: 4 envs, entropy 0; B: 8 envs, entropy 0.01}.
* Exploration data: targets 3ert and 1uyd, exploration seeds {100, 101}, budget B = 2000. Score of a configuration = mean over the 4
  (target, seed) cells of symlog(best_energy / 100 kJ/mol). Each method keeps its best configuration.
* **Subject selection rule**: the new VQC design (not the original) with the lowest score among the new designs is the *subject*. If the original
  `ppo_vqc` scores better than every new design, the subject is the original (and that is reported).

## Phase B - confirmation (the only place a claim can be made)

* Fresh seeds {200..209} (disjoint from exploration {100,101} and from the pilot {0..5}); all three targets (3ert, 1uyd, 1eve); B = 2000.
* Compared: random_search, hill_climb, cmaes (hyper-parameters from the pilot's 3-trial tuning, results/tuning_pilot.json - search methods have no PPO
  settings, so their tuning effort is 3 trials), ppo_mlp_matched, ppo_mlp_large, ppo_mlp_matched_reup (PPO methods: configurations chosen in Phase A),
  the original `ppo_vqc` and the subject.
* **V2 verdict** (mechanical, same operationalisation as PREREGISTRATION.md section "Operationalisation"): the subject is "better than classical
  baselines" only if it beats the best of {random_search, hill_climb, cmaes, ppo_mlp_matched, ppo_mlp_large, ppo_mlp_matched_reup} on >= 2 targets
  (median best energy or success rate, 95% bootstrap CI over seeds for the difference excluding 0 in its favour) AND the Qiskit-Aer MPS simulation
  with bond dimension chi = 4 does not reproduce its outputs (mean |<Z>_MPS - <Z>_exact| over visited states > 0.05; for the re-uploading
  circuit the data-dependent angles are folded into a concrete circuit per input).
* **Secondary, reported separately (not a claim of superiority)**: "parity with classical RL" = the 95% CI of (best PPO-MLP energy - subject energy)
  contains 0 on every target, i.e. the subject is statistically indistinguishable from the best PPO-MLP.
* Anything else prints `NO DEMONSTRATED ADVANTAGE`. Exploration-phase results are never quoted as evidence of an advantage.

## Honest limits stated in advance

* Hill climbing / CMA-ES are strong in 6-8 dimensions; a design that reaches parity with PPO-MLP but not hill climbing will be reported as exactly that.
* The re-uploading design has ~690 trainable parameters, ~20x the original; parameter counts are always reported next to results, and a size-matched MLP is included.
* All quantum results are exact classical simulation; nothing is run on hardware.
