#!/usr/bin/env bash
# Pilot chain (DECISIONS D21): B=2000, 6 seeds, 3 targets, 3-trial tuning. Resumable: finished runs are cached under results/pilot/rows.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
E=pilot
$PY scripts/run_experiments.py --experiment $E --tune
$PY scripts/barren_plateau.py --experiment $E
$PY scripts/noise_eval.py --experiment $E
$PY scripts/simulability.py --experiment $E
$PY scripts/transfer.py --experiment $E
$PY scripts/run_active_learning.py --experiment $E
$PY scripts/run_hardware_eval.py --experiment $E --n-states 5
$PY scripts/benchmark_oracle.py
$PY scripts/make_report.py --experiment $E
echo PILOT_DONE
