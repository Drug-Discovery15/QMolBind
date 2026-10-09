"""Gymnasium torsion-search environment (ligand in a frozen protein pocket)."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qumolbind.env.state import build_state, symlog
from qumolbind.sim.oracle_fast import EnergyTerms, FastOracle
from qumolbind.sim.target import Target


class PoseEnv(gym.Env):
    """Action: K torsion deltas in [-1,1] * max_delta_deg. Episode: T steps from random torsions.

    Reward_t = -(e_t - e_{t-1}) / reward_scale, clipped to +-reward_clip, where e = symlog(score/energy_scale)
    (``energy_transform='symlog'``, default; clash energies reach 1e9 kJ/mol) or e = score (``'linear'``).
    ``score`` = E_int + MMFF strain. Every oracle evaluation (reset included) increments ``oracle.calls``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        target: Target,
        oracle: FastOracle,
        max_delta_deg: float = 60.0,
        episode_len: int = 20,
        state_dim: int = 256,
        reward_clip: float = 10.0,
        reward_scale: float = 1.0,
        energy_scale: float = 100.0,
        energy_transform: str = "symlog",
        lig_emb: np.ndarray | None = None,
        pocket_emb: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.target, self.oracle, self.lm = target, oracle, target.ligand
        self.K = self.lm.K
        self.max_delta, self.T, self.d = max_delta_deg, episode_len, state_dim
        self.reward_clip, self.reward_scale, self.energy_scale = reward_clip, reward_scale, energy_scale
        self.energy_transform = energy_transform
        self.lig_emb, self.pocket_emb = lig_emb, pocket_emb
        self.action_space = spaces.Box(-1.0, 1.0, (self.K,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.d,), dtype=np.float32)
        self.coords = self.lm.native.copy()
        self.t = 0
        self.n_steps = 0
        self.n_clipped = 0
        self._e = 0.0
        self.best_score = np.inf
        self.best_coords = self.coords.copy()
        self.last_terms: EnergyTerms | None = None
        self._calls0 = 0
        self.n_oracle = 0  # this env's own cumulative oracle calls (the oracle may be shared / wrapped)

    # ------------------------------------------------------------------
    def _transform(self, score: float) -> float:
        return float(symlog(score, self.energy_scale)) if self.energy_transform == "symlog" else float(score)

    def _obs(self) -> np.ndarray:
        assert self.last_terms is not None
        return build_state(
            self.lm.get_torsions_deg(self.coords), self.last_terms.terms(), self.t, self.T, self.d,
            self.lig_emb, self.pocket_emb, self.energy_scale,
        )

    def _score(self, coords: np.ndarray) -> EnergyTerms:
        et = self.oracle.evaluate(coords)
        self.n_oracle += 1
        if et.score < self.best_score:
            self.best_score, self.best_coords = et.score, coords.copy()
        return et

    def _info(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.last_terms is not None
        info = {
            "energy": self.last_terms.score,
            "e_int": self.last_terms.e_int,
            "rmsd_native": self.target.rmsd(self.coords),
            "oracle_calls": self.n_oracle - self._calls0,  # per-episode; global total = oracle.calls
            "best_energy": self.best_score,
            "best_rmsd_native": self.target.rmsd(self.best_coords),
        }
        if extra:
            info.update(extra)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.coords = self.lm.randomize(self.np_random)
        self.t = 0
        self._calls0 = self.n_oracle
        self.best_score = np.inf
        self.last_terms = self._score(self.coords)
        self._e = self._transform(self.last_terms.score)
        return self._obs(), self._info()

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.coords = self.lm.apply_torsion_deltas(self.coords, a * self.max_delta)
        self.last_terms = self._score(self.coords)
        e_new = self._transform(self.last_terms.score)
        raw = -(e_new - self._e) / self.reward_scale
        reward = float(np.clip(raw, -self.reward_clip, self.reward_clip))
        clipped = abs(raw) > self.reward_clip
        self._e = e_new
        self.t += 1
        self.n_steps += 1
        self.n_clipped += int(clipped)
        truncated = self.t >= self.T
        return self._obs(), reward, False, truncated, self._info({"reward_clipped": bool(clipped)})

    @property
    def clip_rate(self) -> float:
        return self.n_clipped / max(self.n_steps, 1)


def make_env(
    target_id: str, env_cfg: dict | None = None, oracle_kw: dict | None = None, max_torsions: int = 8,
    target=None, oracle=None, wrap_oracle=None,
) -> PoseEnv:
    from qumolbind.sim.target import load_target

    cfg = dict(env_cfg or {})
    target = target or load_target(target_id, max_torsions)
    okw = {
        "platform": cfg.get("oracle_platform", "auto"), "precision": cfg.get("oracle_precision", "mixed"),
        "include_strain": cfg.get("include_strain", True), "minimize_iters": cfg.get("fast_minimize_steps", 0),
    }
    okw.update(oracle_kw or {})
    oracle = oracle or target.make_oracle(**okw)
    if wrap_oracle is not None:
        oracle = wrap_oracle(oracle)
    allowed = {"max_delta_deg", "episode_len", "state_dim", "reward_clip", "reward_scale", "energy_scale", "energy_transform"}
    return PoseEnv(target, oracle, **{k: v for k, v in cfg.items() if k in allowed})
