"""Run methods x targets x seeds under a shared oracle-call budget B (E1 + the VQC ablations E2/E3/E5).

  python scripts/run_experiments.py --experiment smoke --only baselines   # Stage 4: results/baselines_smoke.csv
  python scripts/run_experiments.py --experiment smoke                    # E1 + E2 + E3 + E5 method pool
  python scripts/run_experiments.py --experiment smoke --with-al          # + active-learning loop (Stage 7)
  --tune : 3 trials per method on separate tuning seeds (equal tuning effort) -> results/tuning_<exp>.json

Every (target, method, seed) run is cached as results/<exp>/rows/*.json (+ curves/*.npy, ckpt/*.pt for VQC) so interrupted
runs resume and `make report` only reads files.
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
E1 = [*BASELINES, "ppo_vqc", "ppo_vqc_proj", "ppo_mlp_matched_proj"]
E2 = ["ppo_vqc", "ppo_vqc_noent"]
E3 = ["ppo_vqc", "ppo_vqc_angle", "ppo_vqc_ryrz", "ppo_vqc_cz"]
E5 = ["ppo_vqc", "ppo_vqc_shots4096", "ppo_vqc_shots1024", "ppo_vqc_shots256"]
POOL = list(dict.fromkeys([*E1, *E2, *E3, *E5]))


def registry() -> dict:
    from qumolbind.quantum.runner import VARIANTS, make_variant_runner

    reg = dict(REGISTRY)
    for name in VARIANTS:
        reg[name] = make_variant_runner(name)
        TUNING_GRID.setdefault(name, TUNING_GRID["ppo_vqc"])
    reg["ppo_mlp_matched_proj"] = _matched_proj
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


def tune(method: str, problem, budget: int, tune_seeds: list[int], reg: dict, actor_cfg: dict) -> dict:
    name, vals = TUNING_GRID[method]
    scores = {}
    for v in vals:
        hp = {} if name == "tune_seed" else {name: v}
        off = v if name == "tune_seed" else 0
        runs = [call(reg, method, problem, budget, s + off, actor_cfg, hp) for s in tune_seeds]
        scores[str(v)] = float(np.median([r.best_score for r in runs]))
    best = min(scores, key=scores.get)
    return {"axis": name, "scores": scores, "best": best, "hparams": {} if name == "tune_seed" else {name: type(vals[0])(best)}}


def call(reg: dict, method: str, problem, budget: int, seed: int, actor_cfg: dict, hp: dict, ckpt: Path | None = None, log_dir: Path | None = None):
    extra = {"actor_cfg": actor_cfg} if method.startswith("ppo") else {}
    if ckpt is not None and (method.startswith("ppo")):
        extra["ckpt_path"] = str(ckpt)
    if log_dir is not None and method.startswith("ppo") and method != "ppo_mlp_matched_proj":
        extra["log_dir"] = str(log_dir)
    return reg[method](problem, budget, seed, **extra, **hp)


def load_rows(rows_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(rows_dir.glob("*.json"))]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--only", default="all", choices=["all", "baselines"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--with-al", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute cached runs")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = to_dict(load_config([f"experiment={a.experiment}", *a.overrides]))
    exp, env_cfg, actor_cfg = d["experiment"], d["env"], d["actor"]
    reg = registry()
    methods = a.methods or (BASELINES if a.only == "baselines" else POOL)
    targets = a.targets or exp["targets"]
    B, seeds = exp["budget"], a.seeds if a.seeds is not None else exp["seeds"]
    out_dir = ROOT / "results" / exp["name"]
    rows_dir = out_dir / "rows"
    for sub in ("curves", "rows", "ckpt"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    tuning_path = ROOT / "results" / f"tuning_{exp['name']}.json"
    tuning: dict = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}

    for tid in targets:
        problem = make_problem(tid, env_cfg)
        print(f"== target {tid}: K={problem.target.ligand.K}, oracle platform={problem.oracle.platform_name}, B={B}", flush=True)
        for m in methods:
            hp = default_hparams(m)
            if a.tune:
                key = f"{tid}/{m}"
                if key not in tuning:
                    tuning[key] = tune(m, problem, B, exp.get("tune_seeds", [1000]), reg, actor_cfg)
                    tuning_path.write_text(json.dumps(tuning, indent=2))
                hp = tuning[key]["hparams"]
            for s in seeds:
                rp = rows_dir / f"{tid}_{m}_{s}.json"
                if rp.exists() and not a.force:
                    continue
                t0 = time.time()
                r = call(reg, m, problem, B, s, actor_cfg, hp, ckpt=out_dir / "ckpt" / f"{tid}_{m}_{s}.pt", log_dir=out_dir / "logs" / f"{tid}_{m}_{s}")
                np.save(out_dir / "curves" / f"{tid}_{m}_{s}.npy", np.stack([r.curve, r.rmsd_curve]))
                row = r.row()
                row["tuned"] = bool(a.tune)
                rp.write_text(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}))
                print(f"  {m:22s} seed={s} best={r.best_score:12.2f} rmsd={r.best_rmsd:5.2f} success={r.success} "
                      f"params={r.n_params} calls={r.calls_used} ({time.time()-t0:.1f}s)", flush=True)

    df = load_rows(rows_dir)
    out = Path(a.out) if a.out else ROOT / "results" / (f"baselines_{exp['name']}.csv" if a.only == "baselines" else f"experiments_{exp['name']}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    (df[df.method.isin(BASELINES)] if a.only == "baselines" else df).to_csv(out, index=False)
    print(f"wrote {out}")

    if a.with_al:
        from scripts.run_active_learning import run_al  # noqa: E402

        for tid in targets:
            for s in seeds[:1]:
                o = run_al({"experiment": exp, "env": env_cfg, "actor": actor_cfg}, tid, s)
                print(f"AL {tid} seed={s}: L2 labels={o['n_L2']} spearman(L1,L2)={o['spearman_L1_vs_L2']:.2f}")


if __name__ == "__main__":
    main()
