"""Reproduce the affinity RMSE table from logs (results/affinity/*/metrics.json); optionally train first."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.encoders.affinity_train import run_target, table_from_logs  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[], help="targets to (re)train, e.g. 1cil 3ert 1uyd")
    ap.add_argument("--kind", default="dimenet")
    ap.add_argument("--max-molecules", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    for t in a.train:
        m = run_target(t, a.kind, a.max_molecules, a.epochs)
        print(t, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}, flush=True)
    tab = table_from_logs()
    (ROOT / "results" / "affinity_table.md").write_text(tab + "\n")
    print(tab)
