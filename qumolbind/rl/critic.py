"""Classical critic (always classical, same state as the actor)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.orthogonal_(lin.weight, np.sqrt(2))
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = h
        out = nn.Linear(d, 1)
        nn.init.orthogonal_(out.weight, 1.0)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
