"""Run methods x targets x seeds under a shared oracle-call budget B and write results/*.csv (+ curves).

  python scripts/run_experiments.py --experiment smoke --only baselines   # Stage 4: results/baselines_smoke.csv
  python scripts/run_experiments.py --experiment smoke                    # all registered methods (E1)
  --tune  : 3 trials per method on separate tuning seeds (equal tuning effort), stored in results/tuning_<exp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines import REGISTRY, TUNING_GRID  # noqa: E402
from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402

BASELINES = ["random_search", "hill_climb", "cmaes", "ppo_mlp_matched", "ppo_mlp_large"]


def registry() -> dict:
    reg = dict(REGISTRY)
    from qumolbind.quantum.runner import VARIANTS, make_variant_runner

    for name in VARIANTS:
        reg[name] = make_variant_runner(name)
        TUNING_GRID.setdefault(name, TUNING_GRID["ppo_vqc"])
    reg["ppo_mlp_matched_proj"] = lambda problem, budget, seed, actor_cfg=None, **kw: _matched_proj(problem, budget, seed, actor_cfg, **kw)
    TUNING_GRID.setdefault("ppo_mlp_matched_proj", TUNING_GRID["ppo_mlp_matched"])
    return reg


def _matched_proj(problem, budget, seed, actor_cfg=None, **kw):
    """MLP matched in size to VQC variant (ii) (trainable 256->256 projection)."""
    from qumolbind.baselines.ppo_mlp import run_ppo_mlp_matched

    res = run_ppo_mlp_matched(problem, budget, seed, actor_cfg={**(actor_cfg or {}), "input_projection": True}, **kw)
    res.method = "ppo_mlp_matched_proj"
    return res


def default_hparams(method: str) -> dict:
    name, vals = TUNING_GRID[method]
    return {} if name == "tune_seed" else {name: vals[len(vals) // 2]}


def tune(method: str, problem, budget: int, tune_seeds: list[int], reg: dict) -> dict:
    name, vals = TUNING_GRID[method]
    scores = {}
    for v in vals:
        hp = {} if name == "tune_seed" else {name: v}
        seed_off = v if name == "tune_seed" else 0
        runs = [reg[method](problem, budget, s + seed_off, **hp) for s in tune_seeds]
        scores[str(v)] = float(np.median([r.best_score for r in runs]))
    best = min(scores, key=scores.get)
    return {"axis": name, "scores": scores, "best": best, "hparams": {} if name == "tune_seed" else {name: type(vals[0])(best)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--only", default="all", choices=["all", "baselines"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config([f"experiment={a.experiment}", *a.overrides])
    d = to_dict(cfg)
    exp, env_cfg = d["experiment"], d["env"]
    reg = registry()
    methods = a.methods or (BASELINES if a.only == "baselines" else [m for m in [*BASELINES, "ppo_vqc"] if m in reg])
    targets = a.targets or exp["targets"]
    B, seeds = exp["budget"], exp["seeds"]
    out_dir = ROOT / "results" / exp["name"]
    (out_dir / "curves").mkdir(parents=True, exist_ok=True)
    rows = []
    tuning: dict = {}
    for tid in targets:
        problem = make_problem(tid, env_cfg)
        print(f"== target {tid}: K={problem.target.ligand.K}, oracle platform={problem.oracle.platform_name}", flush=True)
        for m in methods:
            hp = default_hparams(m)
            if a.tune:
                tuning[f"{tid}/{m}"] = tune(m, problem, B, exp.get("tune_seeds", [1000]), reg)
                hp = tuning[f"{tid}/{m}"]["hparams"]
            for s in seeds:
                t0 = time.time()
                r = reg[m](problem, B, s, actor_cfg=d["actor"], ppo_kw=None, **hp) if m.startswith("ppo") else reg[m](problem, B, s, **hp)
                np.save(out_dir / "curves" / f"{tid}_{m}_{s}.npy", np.stack([r.curve, r.rmsd_curve]))
                rows.append(r.row())
                print(f"  {m:16s} seed={s} best={r.best_score:10.2f} rmsd={r.best_rmsd:5.2f} success={r.success} "
                      f"params={r.n_params} calls={r.calls_used} ({time.time()-t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out) if a.out else ROOT / "results" / (f"baselines_{exp['name']}.csv" if a.only == "baselines" else f"experiments_{exp['name']}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if tuning:
        (ROOT / "results" / f"tuning_{exp['name']}.json").write_text(json.dumps(tuning, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
