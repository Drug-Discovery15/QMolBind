"""Stage 7 driver: QPPO + surrogate/UCB + Level-2 MD loop. Writes results/al/<exp>/<target>_seed<k>/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.active.loop import ActiveLearningLoop  # noqa: E402
from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402


def run_al(cfg: dict, target: str, seed: int, actor_kind: str = "vqc") -> dict:
    exp = cfg["experiment"]
    prob = make_problem(target, cfg["env"])
    loop = ActiveLearningLoop(prob, exp["budget"], exp["al"], cfg["actor"], cfg["actor"].get("lr", 1e-2), seed,
                              ROOT / "results" / "al" / exp["name"] / f"{target}_seed{seed}", actor_kind=actor_kind)
    return loop.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--overrides", nargs="*", default=[])
    a = ap.parse_args()
    cfg = to_dict(load_config([f"experiment={a.experiment}", *a.overrides]))
    for t in a.targets or cfg["experiment"]["targets"]:
        for s in a.seeds if a.seeds is not None else cfg["experiment"]["seeds"][:1]:
            out = run_al(cfg, t, s)
            r = out["result"]
            print(f"AL {t} seed={s}: best_L1={r.best_score:.2f} rmsd={r.best_rmsd:.2f} L2 labels={out['n_L2']} "
                  f"spearman(L1,L2)={out['spearman_L1_vs_L2']:.2f}")
