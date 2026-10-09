"""Global seeding for python / numpy / torch."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed python, numpy and torch (if importable); return a fresh numpy Generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, *keys: int) -> int:
    """Deterministically derive a child seed (e.g. per worker / per env)."""
    ss = np.random.SeedSequence([base_seed, *keys])
    return int(ss.generate_state(1)[0])
