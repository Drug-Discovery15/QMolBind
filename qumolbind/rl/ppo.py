"""CleanRL-style PPO (continuous actions, GAE) used unchanged for every learned policy; only the actor differs.

* actions are sampled from N(mean(obs), exp(log_std)) and stored unclipped (the env clips to [-1,1])
* time-limit truncation is bootstrapped (reward += gamma * V(final_obs)) rather than treated as terminal
* rewards are normalised by the running std of discounted returns (clash energies make raw reward scale wild)
* training stops cleanly when the shared oracle-call budget is exhausted (``BudgetExhausted``)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from qumolbind.rl.budget import BudgetExhausted
from qumolbind.rl.actors import GaussianActor
from qumolbind.rl.buffers import RolloutBuffer, RunningMeanStd
from qumolbind.rl.critic import Critic


@dataclass
class PPOConfig:
    n_envs: int = 4
    rollout_steps: int = 20
    epochs: int = 4
    minibatches: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 1e-2
    critic_lr: float = 1e-3
    norm_reward: bool = True
    norm_adv: bool = True
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class PPO:
    def __init__(
        self, vec_env, actor: GaussianActor, critic: Critic, cfg: PPOConfig,
        logger=None, on_update: Callable[["PPO", dict], None] | None = None,
    ) -> None:
        self.env, self.actor, self.critic, self.cfg, self.logger, self.on_update = vec_env, actor, critic, cfg, logger, on_update
        self.obs_dim = int(vec_env.single_observation_space.shape[0])
        self.act_dim = int(vec_env.single_action_space.shape[0])
        mean_ids = {id(p) for p in actor.mean_params()}
        groups = [
            {"params": [p for p in actor.parameters() if id(p) in mean_ids], "lr": cfg.actor_lr},
            {"params": [actor.log_std], "lr": cfg.actor_lr},
            {"params": list(critic.parameters()), "lr": cfg.critic_lr},
        ]
        self.opt = torch.optim.Adam(groups, eps=1e-5)
        self.ret_rms = RunningMeanStd()
        self.discounted = np.zeros(cfg.n_envs)
        self.history: list[dict] = []
        self.updates = 0
        self.global_steps = 0
        self._obs: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _collect(self, buf: RolloutBuffer, ep_stats: list[dict]) -> None:
        cfg = self.cfg
        for t in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32)
            with torch.no_grad():
                dist = self.actor.dist(obs_t)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = self.critic(obs_t)
            next_obs, rew, term, trunc, infos = self.env.step(act.numpy())
            rew = rew.astype(np.float64)
            done = np.logical_or(term, trunc)
            for i, info in enumerate(infos):  # bootstrap time-limit truncations
                if trunc[i] and not term[i] and "final_observation" in info:
                    with torch.no_grad():
                        fv = self.critic(torch.as_tensor(info["final_observation"], dtype=torch.float32)[None]).item()
                    rew[i] += cfg.gamma * fv
                if done[i]:
                    ep_stats.append({k: info[k] for k in ("energy", "rmsd_native", "best_energy") if k in info})
            if cfg.norm_reward:
                self.discounted = self.discounted * cfg.gamma + rew
                self.ret_rms.update(self.discounted)
                self.discounted[done] = 0.0
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + 1e-8), -10, 10)
            buf.obs[t], buf.act[t], buf.logp[t], buf.val[t] = obs_t, act, logp, val
            buf.rew[t] = torch.as_tensor(rew, dtype=torch.float32)
            buf.done[t] = torch.as_tensor(done, dtype=torch.float32)
            self._obs = next_obs
            self.global_steps += cfg.n_envs

    def _update(self, buf: RolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        with torch.no_grad():
            last_v = self.critic(torch.as_tensor(self._obs, dtype=torch.float32))
        buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)
        data = buf.flat()
        self.last_batch_obs = data["obs"]
        n = data["obs"].shape[0]
        mb = max(n // cfg.minibatches, 1)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0, "actor_grad_norm": 0.0}
        n_mb = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, mb):
                idx = perm[s : s + mb]
                dist = self.actor.dist(data["obs"][idx])
                new_logp = dist.log_prob(data["act"][idx]).sum(-1)
                ent = dist.entropy().sum(-1).mean()
                ratio = (new_logp - data["logp"][idx]).exp()
                adv = data["adv"][idx]
                if cfg.norm_adv and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * ((self.critic(data["obs"][idx]) - data["ret"][idx]) ** 2).mean()
                loss = pg - cfg.ent_coef * ent + cfg.vf_coef * v_loss
                self.opt.zero_grad()
                loss.backward()
                grads = [p.grad.detach() for p in self.actor.mean_params() if p.grad is not None]
                gn = torch.sqrt(sum((g**2).sum() for g in grads)) if grads else torch.tensor(0.0)
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["approx_kl"] += ((ratio - 1) - (ratio.log())).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                stats["pg_loss"] += pg.item(); stats["v_loss"] += v_loss.item(); stats["entropy"] += ent.item()
                stats["actor_grad_norm"] += float(gn)
                n_mb += 1
        return {k: v / max(n_mb, 1) for k, v in stats.items()}

    def train(self, max_updates: int | None = None, seed: int | None = None) -> list[dict]:
        """Run until the budget raises BudgetExhausted (or ``max_updates`` updates)."""
        cfg = self.cfg
        self._obs, _ = self.env.reset(seed=cfg.seed if seed is None else seed)
        t0 = time.time()
        while max_updates is None or self.updates < max_updates:
            buf = RolloutBuffer(cfg.rollout_steps, cfg.n_envs, self.obs_dim, self.act_dim)
            ep_stats: list[dict] = []
            try:
                self._collect(buf, ep_stats)
            except BudgetExhausted:
                break
            st = self._update(buf)
            self.updates += 1
            row: dict[str, Any] = {
                "update": self.updates, "steps": self.global_steps, "wall_s": time.time() - t0,
                "mean_reward": float(buf.rew.mean()), "log_std": float(self.actor.log_std.detach().mean()),
                "n_mean_params": self.actor.n_mean_params(), **st,
            }
            if ep_stats:
                for k in ("energy", "rmsd_native", "best_energy"):
                    vals = [e[k] for e in ep_stats if k in e]
                    if vals:
                        row[f"ep_{k}"] = float(np.median(vals))
            if hasattr(self.env, "oracle_calls"):
                row["oracle_calls"] = self.env.oracle_calls
                row["clip_rate"] = self.env.clip_rate()
            if self.on_update is not None:  # may add diagnostics to ``row`` (e.g. entanglement entropy)
                self.on_update(self, row)
            self.history.append(row)
            if self.logger is not None:
                self.logger.log(self.updates, **{k: v for k, v in row.items() if k != "update"})
        return self.history
