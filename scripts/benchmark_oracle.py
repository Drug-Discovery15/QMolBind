"""Measure fast-oracle throughput (evals/s) per target and platform -> results/oracle_benchmark.csv (read by make_report / ARCHITECTURE.md)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.sim.target import load_target  # noqa: E402

CONFIGS = [("OpenCL", "mixed", 1), ("OpenCL", "double", 1), ("CPU", "n/a", 1), ("CPU", "n/a", 8)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["3ert", "1uyd", "1eve"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-cpu", type=int, default=8, help="evaluations for the (slow) CPU platform")
    a = ap.parse_args()
    rows = []
    for tid in a.targets:
        try:
            t = load_target(tid)
        except FileNotFoundError:
            continue
        for plat, prec, threads in CONFIGS:
            try:
                o = t.make_oracle(platform=plat, precision="mixed" if prec == "n/a" else prec, threads=threads)
            except Exception as e:  # noqa: BLE001
                print(f"skip {tid} {plat}: {e}")
                continue
            n = a.n if plat == "OpenCL" else a.n_cpu
            o.benchmark(3)  # warm-up (kernel compilation)
            eps = o.benchmark(n)
            dev = ""
            if plat == "OpenCL":
                ctx = o.ctx_c
                dev = ctx.getPlatform().getPropertyValue(ctx, "DeviceName")
            rows.append({"target": tid, "n_atoms": o.n_prot + o.n_lig, "platform": plat, "precision": prec, "threads": threads if plat == "CPU" else "",
                         "device": dev, "evals_per_s": eps, "n_evals": n})
            print(rows[-1], flush=True)
    out = ROOT / "results" / "oracle_benchmark.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
