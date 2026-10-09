"""E6: gate-noise robustness. Trained ppo_vqc policies are deployed with depolarizing noise p on every 2-qubit gate
(exact density-matrix simulation, quantum/noise.py), identical starts for every p. Evaluation-only (no training with noise).
Columns: best_score / success of the best pose within the evaluation episodes, mean |noisy - noiseless| action mean."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.env.state import symlog  # noqa: E402
from qumolbind.eval.policy_io import load_vqc_actor, rollout  # noqa: E402
from qumolbind.quantum.noise import noisy_expectations  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402

PS = [0.0, 1e-3, 5e-3, 1e-2]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--episodes", type=int, default=6)
    a = ap.parse_args()
    d = to_dict(load_config([f"experiment={a.experiment}"]))
    exp = d["experiment"]
    rows = []
    for tid in exp["targets"]:
        prob = make_problem(tid, d["env"])
        for s in exp["seeds"]:
            ck = ROOT / "results" / exp["name"] / "ckpt" / f"{tid}_ppo_vqc_{s}.pt"
            if not ck.exists():
                print("missing", ck)
                continue
            actor = load_vqc_actor(ck, prob, d["actor"])
            for p in PS:
                def mean_fn(obs, p=p):
                    return noisy_expectations(actor.vqc, obs.to(torch.float64), p).to(torch.float32)

                states, best, rmsd, e0, e1 = rollout(prob, mean_fn, actor.log_std.detach(), a.episodes, seed=10_000 + s)
                with torch.no_grad():
                    ob = torch.as_tensor(states[:64], dtype=torch.float32)
                    mae = float((mean_fn(ob) - actor(ob)).abs().mean())
                rows.append({"target": tid, "seed": s, "p_depol": p, "n_eval_episodes": a.episodes, "median_best_score": float(np.median(best)),
                             "mean_best_rmsd": float(rmsd.mean()), "success_rate": float((rmsd < 2.0).mean()),
                             "mean_symlog_energy_improvement": float(np.mean(symlog(e0) - symlog(e1))), "mean_abs_action_mean_shift": mae})
                print(rows[-1], flush=True)
    out = ROOT / "results" / f"E6_{exp['name']}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
