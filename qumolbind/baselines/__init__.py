"""Classical baselines. Registry maps method name -> run(problem, budget, seed, **hparams) -> RunResult."""
from __future__ import annotations

from typing import Callable

from qumolbind.baselines import cmaes, hill_climb, random_search
from qumolbind.baselines.common import RunResult
from qumolbind.baselines.ppo_mlp import run_ppo_mlp_large, run_ppo_mlp_matched

REGISTRY: dict[str, Callable[..., RunResult]] = {
    "random_search": random_search.run,
    "hill_climb": hill_climb.run,
    "cmaes": cmaes.run,
    "ppo_mlp_matched": run_ppo_mlp_matched,
    "ppo_mlp_large": run_ppo_mlp_large,
}

# One hyper-parameter axis per method, 3 values each ("equal tuning effort": 3 trials per method).
# PPO methods sweep the actor learning rate; search methods sweep their step-size scale. random_search has no
# hyper-parameter, so its 3 "trials" are 3 independent seeds.
TUNING_GRID: dict[str, tuple[str, list]] = {
    "random_search": ("tune_seed", [0, 1, 2]),
    "hill_climb": ("step_deg", [10.0, 20.0, 40.0]),
    "cmaes": ("sigma0", [10.0, 20.0, 40.0]),
    "ppo_mlp_matched": ("lr", [3e-3, 1e-2, 3e-2]),
    "ppo_mlp_large": ("lr", [3e-3, 1e-2, 3e-2]),
    "ppo_vqc": ("lr", [3e-3, 1e-2, 3e-2]),
}
