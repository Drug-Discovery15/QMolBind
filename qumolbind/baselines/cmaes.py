"""CMA-ES (pycma) over the K-dim torsion vector (degrees, periodic), IPOP-style restarts until the budget ends."""
from __future__ import annotations

import time

import cma
import numpy as np

from qumolbind.baselines.common import BudgetExhausted, Problem, RunResult, finalize


def run(problem: Problem, budget: int, seed: int, sigma0: float = 20.0, **hp) -> RunResult:
    rng = np.random.default_rng(seed)
    tr = problem.tracker(budget)
    K = problem.target.ligand.K
    t0 = time.perf_counter()
    popsize = 4 + int(3 * np.log(K))
    restart = 0
    try:
        while True:
            x0 = rng.uniform(-180, 180, K)
            es = cma.CMAEvolutionStrategy(
                x0, sigma0, {"seed": int(rng.integers(1, 2**31 - 1)), "verbose": -9, "popsize": popsize * 2**restart}
            )
            while not es.stop():
                xs = es.ask()
                es.tell(xs, [tr.evaluate(problem.pose(np.asarray(x))).score for x in xs])
            restart += 1
    except BudgetExhausted:
        pass
    return finalize(tr, "cmaes", problem.target_id, seed, 0, {"sigma0": sigma0, "restarts": restart}, time.perf_counter() - t0)
