"""Phase A of docs/PREREGISTRATION_V3.md: elite-restart exploration (hypothesis generation only).

6 trials per PPO method: lr in {3e-3, 1e-2, 3e-2} x elite-restart probability {0.5, 0.9}; PPO setting fixed to each method's V2 choice.
Exploration seeds {300,301}, targets 3ert + 1uyd, B = 2000. Resumable. Writes results/explore3_selection.json.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.env.state import symlog  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402
from scripts.explore_vqc import SETTINGS, registry, run_call  # noqa: E402

LRS = [3e-3, 1e-2, 3e-2]
ELITE = [0.5, 0.9]
METHODS = ["ppo_mlp_matched", "ppo_mlp_large", "ppo_mlp_matched_reup", "ppo_vqc", "ppo_vqc_aff", "ppo_vqc_reup"]
VQC = ["ppo_vqc", "ppo_vqc_aff", "ppo_vqc_reup"]
SEEDS, TARGETS, B = [300, 301], ["3ert", "1uyd"], 2000


def main() -> None:
    v2 = json.loads((ROOT / "results" / "explore_selection.json").read_text())["configs"]
    d = to_dict(load_config(["experiment=pilot"]))
    actor_cfg = {k: v for k, v in d["actor"].items() if k != "lr"}
    out = ROOT / "results" / "explore3" / "rows"
    out.mkdir(parents=True, exist_ok=True)
    reg = {m: fn for m, fn in registry().items() if m in METHODS}
    for tid in TARGETS:
        base = make_problem(tid, d["env"])
        print(f"== explore3 target {tid}", flush=True)
        for name, fn in reg.items():
            skw = SETTINGS[v2[name]["setting"]]
            for lr, p in product(LRS, ELITE):
                prob = replace(base, env_cfg={**base.env_cfg, "elite_start_prob": p})
                for s in SEEDS:
                    f = out / f"{tid}_{name}_lr{lr:g}_p{p:g}_s{s}.json"
                    if f.exists():
                        continue
                    r = run_call(fn, name, prob, s, lr, dict(skw), actor_cfg)
                    f.write_text(json.dumps({"target": tid, "method": name, "lr": lr, "elite_p": p, "seed": s, "best_score": r.best_score,
                                             "best_rmsd": r.best_rmsd, "success": bool(r.success), "n_params": r.n_params}))
                    print(f"  {name:22s} lr={lr:g} p={p:g} seed={s} best={r.best_score:10.1f} ({r.wall_s:.0f}s)", flush=True)

    rows = [json.loads(p.read_text()) for p in out.glob("*.json")]
    sel: dict = {"configs": {}, "scores": {}}
    for name in METHODS:
        best, best_sc, table = None, np.inf, {}
        for lr, p in product(LRS, ELITE):
            cell = [r for r in rows if r["method"] == name and r["lr"] == lr and r["elite_p"] == p]
            if len(cell) < len(SEEDS) * len(TARGETS):
                continue
            sc = float(np.mean([symlog(r["best_score"], 100.0) for r in cell]))
            table[f"lr={lr:g},p={p:g}"] = sc
            if sc < best_sc:
                best, best_sc = {"lr": lr, "elite_p": p, "setting": v2[name]["setting"], "ppo_kw": SETTINGS[v2[name]["setting"]]}, sc
        sel["configs"][name], sel["scores"][name] = best, {"best": best_sc, "all": table}
    sel["subject"] = min(VQC, key=lambda v: sel["scores"][v]["best"])
    (ROOT / "results" / "explore3_selection.json").write_text(json.dumps(sel, indent=2))
    print("\nV3 exploration scores (mean symlog(best/100), lower is better):")
    for n, sc in sorted(sel["scores"].items(), key=lambda kv: kv[1]["best"]):
        print(f"  {n:24s} {sc['best']:.3f}  config={sel['configs'][n]}")
    print("SUBJECT:", sel["subject"])


if __name__ == "__main__":
    main()
