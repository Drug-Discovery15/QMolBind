"""Cross-platform task runner mirroring the Makefile (make is not available on stock Windows).

Usage: python scripts/tasks.py <env|data|test|smoke|baselines|train-q|experiments|report>
Environment variable EXP=smoke|full selects the experiment config (baselines/train-q default: full; experiments default: smoke).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

IMPORT_CHECK = (
    "import qumolbind, numpy, scipy, pandas, h5py, torch, rdkit, openmm, pennylane, qiskit, "
    "gymnasium, cma, lightgbm, hydra, torch_geometric, transformers, MDAnalysis, Bio, pdbfixer; "
    "print('imports ok')"
)


def run(*args: str, check: bool = True) -> int:
    print("+", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=ROOT)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode


def script(name: str, *args: str, check: bool = True) -> int:
    return run(PY, f"scripts/{name}", *args, check=check)


def exp(default: str) -> str:
    return os.environ.get("EXP", default)


def t_env() -> None:
    run(PY, "-m", "pip", "install", "-e", ".[dev,hardware]")
    run(PY, "-c", IMPORT_CHECK)


def t_data() -> None:
    script("fetch_data.py")  # offline-safe: falls back to tests/fixtures and documents the manual download in docs/DATA.md
    script("prep_targets.py", check=False)


def t_test() -> None:
    run(PY, "-m", "pytest", "-x", "-q")


def t_baselines() -> None:
    script("run_experiments.py", "--only", "baselines", "--experiment", exp("full"))


def t_train_q() -> None:
    script("train.py", f"experiment={exp('full')}", *(["--sweep-lr"] if exp("full") == "full" else []))


def t_experiments() -> None:
    e = exp("smoke")
    script("run_experiments.py", "--experiment", e)                    # E1, E2, E3, E5
    script("barren_plateau.py", "--experiment", e)                     # E4
    script("noise_eval.py", "--experiment", e)                         # E6
    script("simulability.py", "--experiment", e)                       # E7
    script("transfer.py", "--experiment", e)                           # E8
    script("hw_cost_report.py")


def t_report() -> None:
    script("make_report.py", "--experiment", exp("smoke"))


def t_smoke() -> None:
    """1 target, B=500, 2 seeds, all methods, 2 AL rounds, report. Needs the OpenCL platform for OpenMM (see DECISIONS D12)."""
    os.environ["EXP"] = "smoke"
    t_data()
    if not (ROOT / "data_cache" / "targets" / "1eve" / "protein.pdb").exists():
        print("[smoke] 1eve not prepared (offline?) -> falling back to the committed 3ert fixture target")
        os.environ["QMB_TARGETS"] = "3ert"
    t_experiments()
    script("run_active_learning.py", "--experiment", "smoke")
    script("run_hardware_eval.py", "--n-states", "5", check=False)
    script("random_rollouts.py", "--targets", "3ert", "--episodes", "5", check=False)
    t_report()


TASKS = {
    "env": t_env, "data": t_data, "test": t_test, "smoke": t_smoke, "baselines": t_baselines,
    "train-q": t_train_q, "experiments": t_experiments, "report": t_report,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        sys.exit(f"usage: tasks.py {{{'|'.join(TASKS)}}}")
    TASKS[sys.argv[1]]()
