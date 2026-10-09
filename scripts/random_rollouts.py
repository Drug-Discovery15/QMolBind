"""Stage 3 DoD: random-policy episodes end-to-end; logs energy / RMSD curves and oracle throughput to results/."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.env.pose_env import make_env  # noqa: E402
from qumolbind.utils.logging import RunLogger  # noqa: E402
from qumolbind.utils.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["1cil", "3ert", "1uyd"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for tid in a.targets:
        seed_everything(a.seed)
        env = make_env(tid)
        lg = RunLogger(ROOT / "results" / "random_rollouts" / tid, tensorboard=False)
        t0 = time.perf_counter()
        gstep = 0
        for ep in range(a.episodes):
            env.reset(seed=a.seed * 1000 + ep)
            for t in range(env.T):
                _, r, _, _, info = env.step(env.action_space.sample())
                lg.log(gstep, episode=ep, t=t, energy=info["energy"], rmsd_native=info["rmsd_native"], reward=r,
                       best_energy=info["best_energy"], oracle_calls=env.oracle.calls)
                gstep += 1
        dt = time.perf_counter() - t0
        eps = env.oracle.calls / dt
        print(f"{tid}: K={env.K} platform={env.oracle.platform_name} oracle calls={env.oracle.calls} "
              f"{eps:.1f} evals/s (incl. env overhead)  reward clip rate={env.clip_rate:.3f}")
        (ROOT / "results" / "random_rollouts" / tid / "throughput.txt").write_text(
            f"target={tid} platform={env.oracle.platform_name} precision=mixed n_atoms={env.oracle.n_prot + env.oracle.n_lig} evals_per_s={eps:.2f}\n"
        )


if __name__ == "__main__":
    main()
