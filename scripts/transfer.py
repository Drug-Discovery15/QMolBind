"""E8: transfer. Pretrain on target A, fine-tune on target B with a smaller budget, vs training on B from scratch with the
same small budget; VQC vs MLP. Both targets must share the number of torsions K (the MLP output layer is K-specific).
Outputs results/E8_<exp>.csv (one row per method x seed x {pretrained, scratch})."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.baselines.ppo_mlp import run_ppo_mlp_large, run_ppo_mlp_matched  # noqa: E402
from qumolbind.quantum.runner import make_variant_runner  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--source", default="3ert")
    ap.add_argument("--dest", default="1uyd")
    ap.add_argument("--ft-frac", type=float, default=0.25)
    a = ap.parse_args()
    d = to_dict(load_config([f"experiment={a.experiment}"]))
    exp, env_cfg, actor_cfg = d["experiment"], d["env"], d["actor"]
    try:
        A, Bp = make_problem(a.source, env_cfg), make_problem(a.dest, env_cfg)
    except FileNotFoundError as e:
        print(f"E8 skipped: {e}")
        sys.exit(0)
    assert A.target.ligand.K == Bp.target.ligand.K, "transfer needs equal K"
    B, B_ft = exp["budget"], max(int(exp["budget"] * a.ft_frac), 50)
    runners = {"ppo_vqc": make_variant_runner("ppo_vqc"), "ppo_mlp_matched": run_ppo_mlp_matched, "ppo_mlp_large": run_ppo_mlp_large}
    rows = []
    for name, run in runners.items():
        for s in exp["seeds"]:
            ck = ROOT / "results" / "E8" / exp["name"] / f"{name}_{a.source}_{s}.pt"
            kw = {"actor_cfg": actor_cfg} if name != "ppo_mlp_large" else {}
            run(A, B, s, ckpt_path=str(ck), **kw)                                         # pretrain on A
            init = torch.load(ck, weights_only=False)
            for mode, st in (("pretrained", init), ("scratch", None)):
                r = run(Bp, B_ft, s + 500, init_state=st, **kw)                             # fine-tune / scratch on B
                rows.append({"method": name, "seed": s, "mode": mode, "source": a.source, "dest": a.dest, "pretrain_budget": B,
                             "finetune_budget": B_ft, "best_score": r.best_score, "best_rmsd": r.best_rmsd, "success": r.success})
                print(rows[-1], flush=True)
    out = ROOT / "results" / f"E8_{exp['name']}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
