"""Greedy local search (Gaussian torsion perturbations) with random restarts."""
from __future__ import annotations

import time

import numpy as np

from qumolbind.baselines.common import BudgetExhausted, Problem, RunResult, finalize


def run(problem: Problem, budget: int, seed: int, step_deg: float = 20.0, patience: int = 40, **hp) -> RunResult:
    rng = np.random.default_rng(seed)
    tr = problem.tracker(budget)
    K = problem.target.ligand.K
    t0 = time.perf_counter()
    try:
        while True:
            x = rng.uniform(-180, 180, K)
            fx = tr.evaluate(problem.pose(x)).score
            fails = 0
            while fails < patience:
                y = x + rng.normal(0, step_deg, K)
                fy = tr.evaluate(problem.pose(y)).score
                if fy < fx:
                    x, fx, fails = y, fy, 0
                else:
                    fails += 1
    except BudgetExhausted:
        pass
    return finalize(tr, "hill_climb", problem.target_id, seed, 0, {"step_deg": step_deg, "patience": patience}, time.perf_counter() - t0)
