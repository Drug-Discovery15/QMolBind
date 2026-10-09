"""Load trained actors from run checkpoints and collect the states they visit (used by E4/E6/E7/E8)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qumolbind.baselines.common import Problem
from qumolbind.env.pose_env import make_env
from qumolbind.env.vec import SyncVecEnv
from qumolbind.quantum.runner import VARIANTS, make_vqc_actor
from qumolbind.rl.actors import GaussianActor
from qumolbind.utils.seeding import derive_seed


def load_vqc_actor(ckpt: str | Path, problem: Problem, actor_cfg: dict, method: str = "ppo_vqc") -> GaussianActor:
    cfg = {**{k: v for k, v in actor_cfg.items() if k != "lr"}, **VARIANTS.get(method, {})}
    actor = make_vqc_actor(problem.env_cfg.get("state_dim", 256), problem.target.ligand.K, cfg)
    actor.load_state_dict(torch.load(ckpt, weights_only=False)["actor"])
    return actor.eval()


def make_vec(problem: Problem, n_envs: int) -> SyncVecEnv:
    """Un-budgeted vector env sharing the problem's oracle (evaluation only)."""
    return SyncVecEnv([lambda: make_env(problem.target_id, problem.env_cfg, target=problem.target, oracle=problem.oracle) for _ in range(n_envs)])


def rollout(problem: Problem, mean_fn, log_std: torch.Tensor, n_envs: int, seed: int, sample: bool = True, torch_seed: int = 0):
    """One episode in each of n_envs envs with actions ~ N(mean_fn(obs), exp(log_std)). Returns (observations, best_score, best_rmsd, start_energy, final_energy).

    Initial states depend only on ``seed`` so different policies / noise levels are compared on identical starts.
    """
    envs = make_vec(problem, n_envs).envs  # stepped directly: SyncVecEnv would auto-reset and overwrite episode-end statistics
    g = torch.Generator().manual_seed(torch_seed)
    res = [e.reset(seed=derive_seed(seed, i)) for i, e in enumerate(envs)]
    obs = np.stack([o for o, _ in res])
    start_e = np.array([i["energy"] for _, i in res])
    T = envs[0].T
    seen = [obs.copy()]
    for _ in range(T):
        with torch.no_grad():
            mean = mean_fn(torch.as_tensor(obs, dtype=torch.float32))
            act = mean + (torch.randn(mean.shape, generator=g) * log_std.exp() if sample else 0.0)
        obs = np.stack([e.step(a)[0] for e, a in zip(envs, act.numpy())])
        seen.append(obs.copy())
    best = [e.best_score for e in envs]
    best_rmsd = [e.target.rmsd(e.best_coords) for e in envs]
    final_e = np.array([e.last_terms.score for e in envs])
    return np.concatenate(seen), np.array(best), np.array(best_rmsd), start_e, final_e
