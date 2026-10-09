"""Phase A of docs/PREREGISTRATION_V2.md: exploratory sweep (hypothesis generation only).

Every PPO method gets the same 6 trials: lr in {3e-3, 1e-2, 3e-2} x shared PPO setting {A: 4 envs/ent 0, B: 8 envs/ent 0.01}.
Exploration seeds {100,101}, targets 3ert + 1uyd, B = 2000. Resumable (rows cached under results/explore/rows).
Writes results/explore_selection.json (best config per method + the selected subject VQC design).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.baselines.ppo_mlp import run_ppo_mlp_large, run_ppo_mlp_matched  # noqa: E402
from qumolbind.env.state import symlog  # noqa: E402
from qumolbind.quantum.runner import VARIANTS, make_variant_runner  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402

LRS = [3e-3, 1e-2, 3e-2]
SETTINGS = {"A": {"n_envs": 4, "ent_coef": 0.0}, "B": {"n_envs": 8, "ent_coef": 0.01}}
NEW_VQC = ["ppo_vqc_aff", "ppo_vqc_reup", "ppo_vqc_reup_l8", "ppo_vqc_reup_ryrz"]
VQC_ALL = ["ppo_vqc", *NEW_VQC]
SEEDS, TARGETS, B = [100, 101], ["3ert", "1uyd"], 2000


def matched_for(variant: str):
    """MLP whose mean-network size is matched to a given VQC design."""
    def run(problem, budget, seed, lr=1e-2, actor_cfg=None, ppo_kw=None, **kw):
        cfg = {**(actor_cfg or {}), **VARIANTS[variant]}
        res = run_ppo_mlp_matched(problem, budget, seed, lr=lr, actor_cfg=cfg, ppo_kw=ppo_kw)
        res.method = f"ppo_mlp_matched_{variant.replace('ppo_vqc_', '')}"
        return res
    return run


def registry() -> dict:
    reg = {"ppo_mlp_matched": run_ppo_mlp_matched, "ppo_mlp_large": run_ppo_mlp_large, "ppo_mlp_matched_reup": matched_for("ppo_vqc_reup")}
    for v in VQC_ALL:
        reg[v] = make_variant_runner(v)
    return reg


def run_call(fn, name, problem, seed, lr, ppo_kw, actor_cfg):
    kw = {"actor_cfg": actor_cfg} if name != "ppo_mlp_large" else {}
    return fn(problem, B, seed, lr=lr, ppo_kw=ppo_kw, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = to_dict(load_config(["experiment=pilot"]))
    actor_cfg = {k: v for k, v in d["actor"].items() if k != "lr"}
    out = ROOT / "results" / "explore" / "rows"
    out.mkdir(parents=True, exist_ok=True)
    reg = registry()
    for tid in TARGETS:
        prob = make_problem(tid, d["env"])
        print(f"== explore target {tid} K={prob.target.ligand.K}", flush=True)
        for name, fn in reg.items():
            for lr, (sname, skw) in product(LRS, SETTINGS.items()):
                for s in SEEDS:
                    f = out / f"{tid}_{name}_lr{lr:g}_{sname}_s{s}.json"
                    if f.exists() and not a.force:
                        continue
                    r = run_call(fn, name, prob, s, lr, dict(skw), actor_cfg)
                    f.write_text(json.dumps({"target": tid, "method": name, "lr": lr, "setting": sname, "seed": s, "best_score": r.best_score,
                                             "best_rmsd": r.best_rmsd, "success": bool(r.success), "n_params": r.n_params, "wall_s": r.wall_s}))
                    print(f"  {name:22s} lr={lr:g} {sname} seed={s} best={r.best_score:10.1f} params={r.n_params} ({r.wall_s:.0f}s)", flush=True)

    rows = [json.loads(p.read_text()) for p in out.glob("*.json")]
    sel: dict = {"configs": {}, "scores": {}}
    for name in reg:
        best, best_sc, table = None, np.inf, {}
        for lr, sname in product(LRS, SETTINGS):
            cell = [r for r in rows if r["method"] == name and r["lr"] == lr and r["setting"] == sname]
            if len(cell) < len(SEEDS) * len(TARGETS):
                continue
            sc = float(np.mean([symlog(r["best_score"], 100.0) for r in cell]))
            table[f"lr={lr:g},{sname}"] = sc
            if sc < best_sc:
                best, best_sc = {"lr": lr, "setting": sname, "ppo_kw": SETTINGS[sname]}, sc
        sel["configs"][name], sel["scores"][name] = best, {"best": best_sc, "all": table}
    new_scores = {v: sel["scores"][v]["best"] for v in NEW_VQC if sel["configs"].get(v)}
    best_new = min(new_scores, key=new_scores.get)
    orig = sel["scores"]["ppo_vqc"]["best"]
    sel["subject"] = best_new if new_scores[best_new] < orig else "ppo_vqc"
    sel["rule"] = "lowest-scoring new design; original if it beats all new designs (docs/PREREGISTRATION_V2.md)"
    (ROOT / "results" / "explore_selection.json").write_text(json.dumps(sel, indent=2))
    print("\nexploration scores (mean symlog(best/100), lower is better):")
    for n, sc in sorted(sel["scores"].items(), key=lambda kv: kv[1]["best"]):
        print(f"  {n:24s} {sc['best']:.3f}  config={sel['configs'][n]}")
    print("SUBJECT:", sel["subject"])


if __name__ == "__main__":
    main()
