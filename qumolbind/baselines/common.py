"""Shared budget accounting for every method (search baselines, PPO-MLP, PPO-VQC)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

SUCCESS_RMSD = 2.0  # Angstrom, evaluated on the best-energy pose (RMSD is evaluation-only)


from qumolbind.rl.budget import BudgetExhausted, Tracker  # noqa: F401  (re-exported)


@dataclass
class RunResult:
    method: str
    target: str
    seed: int
    budget: int
    best_score: float
    best_rmsd: float
    success: bool
    calls_used: int
    n_params: int = 0
    hparams: dict[str, Any] = field(default_factory=dict)
    curve: np.ndarray | None = None
    rmsd_curve: np.ndarray | None = None
    wall_s: float = 0.0

    def row(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("curve", "rmsd_curve", "hparams")}
        d.update({f"hp_{k}": v for k, v in self.hparams.items()})
        return d


def finalize(tracker: Tracker, method: str, target: str, seed: int, n_params: int = 0, hparams: dict | None = None, wall_s: float = 0.0) -> RunResult:
    curve = tracker.curve.copy()
    rc = tracker.rmsd_curve.copy()
    n = tracker.calls
    if n < tracker.budget and n > 0:  # forward-fill (a method that stopped early keeps its best)
        curve[n:], rc[n:] = curve[n - 1], rc[n - 1]
    return RunResult(
        method=method, target=target, seed=seed, budget=tracker.budget, best_score=float(tracker.best_score),
        best_rmsd=float(tracker.best_rmsd), success=bool(tracker.best_rmsd < SUCCESS_RMSD), calls_used=n,
        n_params=n_params, hparams=hparams or {}, curve=curve, rmsd_curve=rc, wall_s=wall_s,
    )


@dataclass
class Problem:
    """One target's shared objects (target geometry, oracle, env config). Reused across methods/seeds."""

    target_id: str
    target: Any
    oracle: Any
    env_cfg: dict[str, Any]

    def tracker(self, budget: int) -> Tracker:
        return Tracker(self.oracle, budget, self.target.rmsd)

    def pose(self, torsions_deg: np.ndarray) -> np.ndarray:
        lm = self.target.ligand
        return lm.set_torsions_deg(lm.native, torsions_deg)


def make_problem(target_id: str, env_cfg: dict[str, Any] | None = None) -> Problem:
    from qumolbind.sim.target import load_target

    cfg = dict(env_cfg or {})
    target = load_target(target_id, cfg.get("max_torsions", 8))
    oracle = target.make_oracle(
        platform=cfg.get("oracle_platform", "auto"), precision=cfg.get("oracle_precision", "mixed"),
        include_strain=cfg.get("include_strain", True), minimize_iters=cfg.get("fast_minimize_steps", 0),
    )
    return Problem(target_id, target, oracle, cfg)


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    import time

    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0
