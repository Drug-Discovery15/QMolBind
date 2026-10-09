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

    def evaluate(self, coords: np.ndarray):
        if self.calls >= self.budget:
            raise BudgetExhausted
        et = self._oracle.evaluate(coords)
        if et.score < self.best_score:
            self.best_score, self.best_coords = et.score, np.array(coords, copy=True)
            self.best_rmsd = self.rmsd_fn(coords)
        self.curve[self.calls] = self.best_score
        self.rmsd_curve[self.calls] = self.best_rmsd
        self.calls += 1
        return et

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def __getattr__(self, name: str):  # forward everything else (protein_positions_nm, params, ...)
        return getattr(self._oracle, name)
