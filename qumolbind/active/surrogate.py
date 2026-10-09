"""Level-0 surrogate: ensemble of K small MLPs (bootstrap-trained) predicting the Level-2 score from cheap pose features.

Features (20): sin/cos of torsions (K_max=8 slots each) + symlog-normalised Level-1 terms (vdW, elec, solv, strain).
No RMSD / native information. Target: symlog(dG / 100 kJ/mol). Uncertainty = ensemble standard deviation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qumolbind.env.state import K_MAX, symlog

FEAT_DIM = 2 * K_MAX + 4


def pose_features(torsions_deg: np.ndarray, terms4: np.ndarray) -> np.ndarray:
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    k = len(torsions_deg)
    r = np.radians(torsions_deg)
    f[:k], f[K_MAX : K_MAX + k] = np.sin(r), np.cos(r)
    f[2 * K_MAX :] = symlog(np.asarray(terms4), 100.0) / 5.0
    return f


def to_target(dg: np.ndarray) -> np.ndarray:
    return symlog(np.asarray(dg, dtype=np.float64), 100.0)


def from_target(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * 100.0 * np.expm1(np.abs(y))


class SurrogateEnsemble:
    def __init__(self, n_members: int = 5, hidden: int = 64, epochs: int = 200, lr: float = 3e-3, weight_decay: float = 1e-3,
                 seed: int = 0) -> None:
        self.n_members, self.hidden, self.epochs, self.lr, self.wd, self.seed = n_members, hidden, epochs, lr, weight_decay, seed
        self.members: list[nn.Module] = []
        self.y_mean, self.y_std = 0.0, 1.0
        self.n_train = 0

    def _net(self) -> nn.Module:
        return nn.Sequential(nn.Linear(FEAT_DIM, self.hidden), nn.Tanh(), nn.Linear(self.hidden, self.hidden), nn.Tanh(), nn.Linear(self.hidden, 1))

    def fit(self, X: np.ndarray, y_raw: np.ndarray) -> "SurrogateEnsemble":
        """y_raw = Level-2 dG in kJ/mol (transformed internally)."""
        y = to_target(y_raw)
        self.n_train = len(y)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-6) if len(y) > 1 else 1.0
        Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for m in range(self.n_members):
            torch.manual_seed(int(rng.integers(1 << 30)))
            idx = torch.as_tensor(rng.integers(0, len(yt), len(yt)))  # bootstrap resample
            net = self._net()
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = ((net(Xt[idx]).squeeze(-1) - yt[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            self.members.append(net.eval())
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mu, sigma) in TARGET (symlog) units. Untrained -> prior N(0, 1)."""
        if not self.members:
            return np.zeros(len(X)), np.ones(len(X))
        with torch.no_grad():
            p = torch.stack([m(torch.as_tensor(X, dtype=torch.float32)).squeeze(-1) for m in self.members]).numpy()
        p = p * self.y_std + self.y_mean
        return p.mean(0), p.std(0)

    def predict_dg(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mu, sigma) mapped back to kJ/mol (sigma = half the +-1 sigma interval after the monotone inverse transform)."""
        mu, sd = self.predict(X)
        return from_target(mu), 0.5 * (from_target(mu + sd) - from_target(mu - sd))
