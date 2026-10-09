"""PPO runners. ``run_ppo`` is shared by the MLP baselines and the VQC policy: only the actor differs."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch

from qumolbind.baselines.common import Problem, RunResult, Tracker, finalize
from qumolbind.env.pose_env import make_env
from qumolbind.env.vec import SyncVecEnv
from qumolbind.quantum.counts import vqc_param_count
from qumolbind.rl.actors import GaussianActor, MLPActor, matched_mlp_hidden, mlp_param_count
from qumolbind.rl.critic import Critic
from qumolbind.rl.ppo import PPO, PPOConfig
from qumolbind.utils.seeding import seed_everything


def build_vec_env(problem: Problem, tracker: Tracker, n_envs: int) -> SyncVecEnv:
    fns = [
        (lambda: make_env(problem.target_id, problem.env_cfg, target=problem.target, oracle=problem.oracle, wrap_oracle=lambda o: tracker))
        for _ in range(n_envs)
    ]
    return SyncVecEnv(fns)


def run_ppo(
    problem: Problem, budget: int, seed: int, method: str, make_actor: Callable[[int, int], GaussianActor],
    lr: float = 1e-2, ppo_kw: dict | None = None, logger=None, hparams: dict | None = None,
    init_state: dict | None = None, ckpt_path: str | None = None, on_update=None, log_dir: str | None = None,
) -> tuple[RunResult, PPO]:
    """``init_state`` = {'actor': sd, 'critic': sd} loads pretrained weights (transfer, E8); ``ckpt_path`` saves final weights."""
    seed_everything(seed)
    kw = dict(ppo_kw or {})
    cfg = PPOConfig(actor_lr=lr, seed=seed, **kw)
    tracker = problem.tracker(budget)
    vec = build_vec_env(problem, tracker, cfg.n_envs)
    d = problem.env_cfg.get("state_dim", 256)
    K = problem.target.ligand.K
    actor = make_actor(d, K)
    critic = Critic(d)
    if init_state is not None:
        actor.load_state_dict(init_state["actor"])
        critic.load_state_dict(init_state["critic"])
    if logger is None and log_dir is not None:
        from qumolbind.utils.logging import RunLogger

        logger = RunLogger(log_dir, "metrics", tensorboard=False)
    ppo = PPO(vec, actor, critic, cfg, logger=logger, on_update=on_update)
    t0 = time.perf_counter()
    ppo.train()
    if ckpt_path is not None:
        import pathlib

        pathlib.Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "target": problem.target_id, "seed": seed, "method": method}, ckpt_path)
    hp = {"lr": lr, "n_mean_params": actor.n_mean_params(), "clip_rate": vec.clip_rate(), **(hparams or {})}
    res = finalize(tracker, method, problem.target_id, seed, actor.n_mean_params(), hp, time.perf_counter() - t0)
    return res, ppo


def matched_target_params(actor_cfg: dict | None, K: int) -> int:
    c = dict(actor_cfg or {})
    return vqc_param_count(
        c.get("n_qubits", 8), c.get("n_layers", 4), c.get("rotations", "ry"), K,
        c.get("trainable_scale", False), c.get("input_projection", False),
    )["total_mean_net"]


def run_ppo_mlp_matched(problem: Problem, budget: int, seed: int, lr: float = 1e-2, actor_cfg: dict | None = None,
                        ppo_kw: dict | None = None, logger=None, ckpt_path=None, init_state=None, log_dir=None, **hp) -> RunResult:
    d, K = problem.env_cfg.get("state_dim", 256), problem.target.ligand.K
    target = matched_target_params(actor_cfg, K)
    hidden = matched_mlp_hidden(d, K, target, 1)
    res, _ = run_ppo(problem, budget, seed, "ppo_mlp_matched", lambda d_, k_: MLPActor(d_, k_, hidden), lr, ppo_kw, logger,
                     {"hidden": str(hidden), "matched_target_params": target}, init_state=init_state, ckpt_path=ckpt_path, log_dir=log_dir)
    return res


def run_ppo_mlp_large(problem: Problem, budget: int, seed: int, lr: float = 1e-2, ppo_kw: dict | None = None,
                      logger=None, ckpt_path=None, init_state=None, log_dir=None, **hp) -> RunResult:
    res, _ = run_ppo(problem, budget, seed, "ppo_mlp_large", lambda d_, k_: MLPActor(d_, k_, (128, 128)), lr, ppo_kw, logger,
                     {"hidden": "(128, 128)"}, init_state=init_state, ckpt_path=ckpt_path, log_dir=log_dir)
    return res
