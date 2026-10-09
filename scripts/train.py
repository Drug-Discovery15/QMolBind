"""Stage 6: train one QPPO (VQC actor) run with full logging. Hydra-style overrides, e.g.
   python scripts/train.py experiment=smoke actor.lr=0.03 seed=1 target=1cil [--sweep-lr]
Logs reward, energy, RMSD, parameter counts, gradient norms, entanglement entropy and wall-clock to results/train_q/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.quantum.runner import quantum_diagnostics, run_ppo_vqc  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402
from qumolbind.utils.logging import RunLogger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--target", default=None)
    ap.add_argument("--sweep-lr", action="store_true", help="run the lr sweep {3e-3,1e-2,3e-2}")
    a = ap.parse_args()
    d = to_dict(load_config(a.overrides))
    exp, env_cfg, actor_cfg = d["experiment"], d["env"], d["actor"]
    target = a.target or exp["targets"][0]
    problem = make_problem(target, env_cfg)
    lrs = [3e-3, 1e-2, 3e-2] if a.sweep_lr else [actor_cfg["lr"]]
    for lr in lrs:
        run_dir = ROOT / "results" / "train_q" / f"{target}_lr{lr:g}_seed{d['seed']}"
        lg = RunLogger(run_dir, "metrics", tensorboard=True)
        t0 = time.time()
        from qumolbind.rl.ppo import PPO

        orig = PPO.__init__

        def patched(self, *args, **kw):
            orig(self, *args, **kw)
            self.on_update = quantum_diagnostics

        PPO.__init__ = patched
        try:
            res = run_ppo_vqc(problem, exp["budget"], d["seed"], lr=lr, actor_cfg=actor_cfg, logger=lg)
        finally:
            PPO.__init__ = orig
        lg.close()
        print(f"target={target} lr={lr:g} seed={d['seed']}: best_energy={res.best_score:.2f} best_rmsd={res.best_rmsd:.2f} "
              f"success={res.success} calls={res.calls_used} mean-net params={res.n_params} "
              f"(quantum={res.hparams.get('n_quantum_params')}, classical={res.hparams.get('n_classical_mean_params')}) "
              f"wall={time.time()-t0:.1f}s  log={run_dir/'metrics.csv'}")


if __name__ == "__main__":
    main()
