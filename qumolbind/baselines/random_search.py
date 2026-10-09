"""Uniform random search over the K-dim torsion vector."""
from __future__ import annotations

import time

import numpy as np

from qumolbind.baselines.common import BudgetExhausted, Problem, RunResult, finalize


def run(problem: Problem, budget: int, seed: int, **hp) -> RunResult:
    rng = np.random.default_rng(seed)
    tr = problem.tracker(budget)
    K = problem.target.ligand.K
    t0 = time.perf_counter()
    try:
        while True:
            tr.evaluate(problem.pose(rng.uniform(-180, 180, K)))
    except BudgetExhausted:
        pass
    return finalize(tr, "random_search", problem.target_id, seed, 0, {}, time.perf_counter() - t0)
