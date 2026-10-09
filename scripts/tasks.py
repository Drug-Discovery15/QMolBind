"""Cross-platform task runner mirroring the Makefile (make is not available on stock Windows).

Usage: python scripts/tasks.py <env|data|test|smoke|baselines|train-q|experiments|report>
"""
from __future__ import annotations

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


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def script(name: str, *args: str) -> None:
    run(PY, f"scripts/{name}", *args)


def t_env() -> None:
    run(PY, "-m", "pip", "install", "-e", ".[dev]")
    run(PY, "-c", IMPORT_CHECK)


def t_data() -> None:
    script("fetch_data.py")


def t_test() -> None:
    run(PY, "-m", "pytest", "-x", "-q")


def t_baselines() -> None:
    script("run_experiments.py", "--only", "baselines", "--experiment", "full")


def t_train_q() -> None:
    script("train.py", "experiment=full")


def t_experiments() -> None:
    script("run_experiments.py", "--experiment", "smoke")


def t_report() -> None:
    script("make_report.py")


def t_smoke() -> None:
    t_data()
    script("run_experiments.py", "--experiment", "smoke", "--with-al")
    t_report()


TASKS = {
    "env": t_env, "data": t_data, "test": t_test, "smoke": t_smoke, "baselines": t_baselines,
    "train-q": t_train_q, "experiments": t_experiments, "report": t_report,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        sys.exit(f"usage: tasks.py {{{'|'.join(TASKS)}}}")
    TASKS[sys.argv[1]]()
