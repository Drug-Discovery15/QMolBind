"""Rollout storage + GAE."""
from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, steps: int, n_envs: int, obs_dim: int, act_dim: int) -> None:
        self.steps, self.n_envs = steps, n_envs
        self.obs = torch.zeros(steps, n_envs, obs_dim)
        self.act = torch.zeros(steps, n_envs, act_dim)
        self.logp = torch.zeros(steps, n_envs)
        self.rew = torch.zeros(steps, n_envs)
        self.done = torch.zeros(steps, n_envs)   # episode ended after this step (term or trunc)
        self.val = torch.zeros(steps, n_envs)
        self.adv = torch.zeros(steps, n_envs)
        self.ret = torch.zeros(steps, n_envs)

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        last_gae = torch.zeros(self.n_envs)
        for t in reversed(range(self.steps)):
            next_val = last_value if t == self.steps - 1 else self.val[t + 1]
            nonterminal = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * next_val * nonterminal - self.val[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            self.adv[t] = last_gae
        self.ret = self.adv + self.val

    def flat(self) -> dict[str, torch.Tensor]:
        n = self.steps * self.n_envs
        return {
            "obs": self.obs.reshape(n, -1), "act": self.act.reshape(n, -1), "logp": self.logp.reshape(n),
            "adv": self.adv.reshape(n), "ret": self.ret.reshape(n), "val": self.val.reshape(n),
        }


class RunningMeanStd:
    def __init__(self) -> None:
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = float(np.mean(x)), float(np.var(x)), x.size
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot
