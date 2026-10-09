# Pre-registered success criterion (written before any E1 run)

Committed to git before E1 was executed; `scripts/make_report.py` evaluates it mechanically from `results/`.

## Verbatim criterion (from the project brief)

> "The VQC policy is reported as better than classical baselines only if, at oracle budget B on >= 2 targets, its median
> success rate (or median best energy) beats the best classical baseline with a 95% bootstrap CI for the difference that
> excludes 0, **and** E7 shows MPS simulation at small chi does not reproduce its outputs. Otherwise report: no
> demonstrated advantage."

## Operationalisation (fixed now, not tunable afterwards)

* **VQC policy** = `ppo_vqc` (variant (i): amplitude encoding, RY + CNOT ring, n=8, L=4, no trainable projection),
  with the learning rate chosen by the same 3-trial tuning protocol as every other method. Other VQC variants
  (`ppo_vqc_proj`, ablations) are reported but are NOT the subject of the criterion.
* **Classical baselines** = `random_search`, `hill_climb`, `cmaes`, `ppo_mlp_matched`, `ppo_mlp_large`
  (`ppo_mlp_matched_proj` is reported as an extra; it is not part of the "best classical baseline" set to keep the set
  fixed to the five prescribed ones).
* **Budget B** = the experiment config's total oracle-call budget (smoke: 500, full: 20000), the same for all methods.
* **Per target, per seed** metrics: `best_score` (lowest Level-1 score reached within B calls; lower is better) and
  `success` (RMSD of the best-score pose < 2 A). Each metric is aggregated over seeds: median best energy; success rate =
  fraction of successful seeds.
* **Comparison**: for each metric the "best classical baseline" is the baseline with the best point estimate of that metric
  on that target (selected on the same data, which favours the baselines = conservative for the VQC).
  Difference = baseline_median_energy - vqc_median_energy (positive favours the VQC) or vqc_rate - baseline_rate.
  The 95% CI is a percentile bootstrap over seeds (10,000 resamples, resampling each group's seeds independently).
  The VQC "wins" a metric on a target iff the whole CI is > 0.
* **Target-level win**: the VQC wins on energy OR on success rate (the brief allows either; no multiplicity correction is
  applied, which is lenient towards the VQC and is stated here so a win must be read with that in mind).
* **>= 2 targets**: at least two distinct targets are target-level wins.
* **E7 clause**: on the trained `ppo_vqc` policies (final weights, all seeds/targets pooled), the Qiskit-Aer MPS simulation with
  bond dimension chi = 4 (small chi; max exact chi for n=8 is 16) must NOT reproduce the exact output:
  mean |<Z>_MPS(chi=4) - <Z>_exact| over the policy's visited states **> 0.05**. If the error is <= 0.05 the circuit is
  cheaply classically simulable and the clause fails.
* **Verdict**: `VQC reported as better than classical baselines` iff (>= 2 target-level wins) AND (E7 clause holds).
  Anything else, including missing data, fewer than 2 targets, or E7 not run, prints `NO DEMONSTRATED ADVANTAGE`.
  The report states which condition failed. Smoke settings (2 seeds, B=500) cannot satisfy this criterion by design
  (a bootstrap over 2 seeds has no power); the smoke report therefore shows the branch only to prove the logic runs.

## Other fixed analysis choices

* Pairwise comparisons additionally report Mann-Whitney U (two-sided) with the rank-biserial correlation as effect size.
* Curves are median +- IQR across seeds; the best seed is never reported alone.
* No claim of superiority over classical methods is made anywhere in the repository unless this criterion's positive branch is met.
