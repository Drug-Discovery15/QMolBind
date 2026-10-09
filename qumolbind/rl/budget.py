"""Oracle-call budget accounting shared by all methods."""
from __future__ import annotations

from typing import Any

import numpy as np

class BudgetExhausted(Exception):
    """Raised by the Tracker when the total oracle-call budget B is used up."""


class Tracker:
    """Wraps an oracle: hard-caps total calls at ``budget`` and records the best-energy-so-far curve.

    Shared by all environments / workers of a single run so that *every* call (including env resets) is counted.
    """

    def __init__(self, oracle: Any, budget: int, rmsd_fn) -> None:
        self._oracle = oracle
        self.budget = budget
        self.rmsd_fn = rmsd_fn
        self.calls = 0
        self.best_score = np.inf
        self.best_coords: np.ndarray | None = None
        self.best_rmsd = np.nan
        self.curve = np.full(budget, np.nan)       # best score after call i
        self.rmsd_curve = np.full(budget, np.nan)  # RMSD of best-score pose after call i
        self._nf0 = getattr(oracle, "n_nonfinite", 0)
        self.log_poses = False
        self.elite_k = 20
        self.archive: list[tuple[float, np.ndarray]] = []  # best poses seen so far, sorted by score (used by elite-restart episodes)
        self.pose_log: list[tuple[np.ndarray, np.ndarray]] = []  # (coords, [vdw, elec, solv, strain, score]) when enabled

    def evaluate(self, coords: np.ndarray):
        if self.calls >= self.budget:
            raise BudgetExhausted
        et = self._oracle.evaluate(coords)
        if et.score < self.best_score:
            self.best_score, self.best_coords = et.score, np.array(coords, copy=True)
            self.best_rmsd = self.rmsd_fn(coords)
        if len(self.archive) < self.elite_k or et.score < self.archive[-1][0]:
            self.archive.append((float(et.score), np.array(coords, copy=True)))
            self.archive.sort(key=lambda t: t[0])
            del self.archive[self.elite_k :]
        if self.log_poses:
            self.pose_log.append((np.array(coords, copy=True), np.array([et.vdw, et.elec, et.solv, et.strain, et.score])))
        self.curve[self.calls] = self.best_score
        self.rmsd_curve[self.calls] = self.best_rmsd
        self.calls += 1
        return et

    @property
    def n_nonfinite(self) -> int:
        """NaN/inf oracle energies replaced by a finite cap during this tracker's lifetime."""
        return getattr(self._oracle, "n_nonfinite", 0) - self._nf0

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def __getattr__(self, name: str):  # forward everything else (protein_positions_nm, params, ...)
        return getattr(self._oracle, name)
