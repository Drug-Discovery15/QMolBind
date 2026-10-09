# Pre-registration V5: high-power replication of V4 (written before any V5 run)

V4 (REPORT_V4.md, 10 seeds) ended with `NO DEMONSTRATED ADVANTAGE` and `NOT PARITY`, but the VQC (`ppo_vqc_aff`) was within ~70-80 kJ/mol of the best
classical method on all three targets and the 95% intervals were wide (lower ends -329, -238, -244 kJ/mol), so low power may explain the failed
non-inferiority test. V5 changes **nothing** about the algorithms, hyper-parameters, subject, baselines, budget, margin or verdict rules of
docs/PREREGISTRATION_V4.md; it only enlarges the sample.

* Same configuration files: results/explore4_selection.json (subject `ppo_vqc_aff` and every method's configuration are taken as selected in V4's exploration;
  nothing is re-tuned and nothing from V4's confirmation seeds is used to choose anything).
* Fresh seeds {700..729} (30 per cell), targets 3ert, 1uyd, 1eve, B = 2000. Search baselines use the pilot's tuned hyper-parameters, as in V4.
* Verdicts exactly as in PREREGISTRATION_V4.md: (1) advantage verdict, (2) parity verdict with margin 100 kJ/mol on every target and the success-rate condition,
  (3) diagnostic against `ppo_zero_mean`.
* V4 and V5 are separate samples and are NOT pooled; V5 is the one that counts for the parity question. Because a sequence of rounds (V2-V5) was run, a single
  positive round would still call for another independent replication before being believed.

If V5 is also `NOT PARITY`, the honest conclusion stands: the redesigned VQC is competitive with classical RL but not demonstrated to match the best classical optimiser.
