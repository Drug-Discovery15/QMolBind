"""Actors for PPO. Contract: ``forward(obs[B,d]) -> mean[B,K]`` in [-1,1]; classical state-independent ``log_std``.

The Gaussian ``log_std`` is ALWAYS classical (quantum = mean network only; see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GaussianActor(nn.Module):
    """Base: subclasses implement ``mean_net``; log_std is a classical learnable vector."""

    def __init__(self, n_actions: int, init_log_std: float = -0.7) -> None:
        super().__init__()
        self.K = n_actions
        self.log_std = nn.Parameter(torch.full((n_actions,), float(init_log_std)))

    def mean_net(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_net(obs)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.forward(obs).to(torch.float32)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def mean_params(self) -> list[nn.Parameter]:
        """Parameters of the mean network (what 'matched parameter count' refers to)."""
        return [p for n, p in self.named_parameters() if n != "log_std"]

    def n_mean_params(self) -> int:
        return int(sum(p.numel() for p in self.mean_params()))

    def n_params_total(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


class MLPActor(GaussianActor):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128), init_log_std: float = -0.7) -> None:
        super().__init__(n_actions, init_log_std)
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.orthogonal_(lin.weight, np.sqrt(2))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = h
        out = nn.Linear(d, n_actions)
        nn.init.orthogonal_(out.weight, 0.01)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        self.hidden = hidden

    def mean_net(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


def mlp_param_count(obs_dim: int, hidden: tuple[int, ...], n_actions: int) -> int:
    dims = [obs_dim, *hidden, n_actions]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


def matched_mlp_hidden(obs_dim: int, n_actions: int, target_params: int, n_hidden_layers: int = 1) -> tuple[int, ...]:
    """Hidden widths whose mean-network parameter count is closest to ``target_params``.

    With the raw d=256 input even a width-1 layer has >256 parameters, so for tiny targets (e.g. a 32-parameter VQC)
    exact matching is infeasible; the closest feasible network is returned and both counts are reported.
    """
    best, best_gap = (1,) * n_hidden_layers, float("inf")
    for h in range(1, 2049):
        hidden = (h,) * n_hidden_layers
        gap = abs(mlp_param_count(obs_dim, hidden, n_actions) - target_params)
        if gap < best_gap:
            best, best_gap = hidden, gap
    return best
