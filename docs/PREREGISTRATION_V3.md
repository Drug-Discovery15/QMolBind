# Pre-registration V3: elite-restart episodes for every PPO policy (written before any V3 run)

Context: V2 (REPORT_V2.md) found the redesigned VQC statistically on par with the best classical PPO policy but still clearly worse than hill climbing
on 2/3 targets. Hypothesis: hill climbing wins because it is a local refinement method, whereas PPO episodes always restart from RANDOM torsions
and the budget (2000 calls = ~95 episodes) is spent mostly escaping clashes. Change tested here (applied to ALL PPO methods, MLP and VQC alike):
**elite-restart episodes** - with probability p an episode starts from one of the 20 best poses found so far in the run (shared oracle-call budget and
tracker; the archive only holds poses the run itself already evaluated, no extra information). Rewards, state, actions and the oracle are unchanged.

## Phase A - exploration (hypothesis generation; no claims)

* PPO designs: `ppo_mlp_matched`, `ppo_mlp_large`, `ppo_mlp_matched_reup`, `ppo_vqc`, `ppo_vqc_aff`, `ppo_vqc_reup`. Each keeps the shared PPO setting (4/8 envs,
  entropy) that V2's exploration selected for it.
* Equal effort: 6 trials per method = lr {3e-3, 1e-2, 3e-2} x elite probability p {0.5, 0.9}. Exploration seeds {300, 301}, targets 3ert and 1uyd, B = 2000.
* Score = mean over the 4 cells of symlog(best_energy / 100 kJ/mol). Each method keeps its best configuration.
* Subject = the VQC design (among `ppo_vqc`, `ppo_vqc_aff`, `ppo_vqc_reup`) with the lowest best score.

## Phase B - confirmation (the only place a claim can be made)

* Fresh seeds {400..409} (disjoint from pilot {0..5}, V2 exploration {100,101}, V2 confirmation {200..209}, V3 exploration {300,301}); 3ert, 1uyd, 1eve; B = 2000.
* Baselines for the criterion: random_search, hill_climb, cmaes (hyper-parameters from the pilot's 3-trial tuning), ppo_mlp_matched, ppo_mlp_large,
  ppo_mlp_matched_reup (all three with elite restarts and their own Phase-A configuration). Subject as above.
* **V3 verdict**: identical mechanical rule to PREREGISTRATION_V2.md (beats the best baseline on >= 2 targets with a 95% bootstrap CI excluding 0 on median
  best energy or success rate, AND Aer-MPS chi=4 error > 0.05 on the subject's visited states). Otherwise `NO DEMONSTRATED ADVANTAGE`.
* Secondary (reported separately): parity with the best PPO-MLP (CI of the energy difference contains 0), and a plain description of whether elite
  restarts narrowed the gap to hill climbing for the MLPs and VQCs alike (reported from the V3 confirmation numbers vs the V2 confirmation numbers,
  which used different seeds - descriptive only).

## Honest limits stated in advance

* Elite restarts are a classical algorithmic change; if they help, they help all PPO policies and say nothing about quantum advantage by themselves.
* A positive verdict would still be an exact classical simulation of 8 qubits on 3 pilot targets, not hardware evidence.
