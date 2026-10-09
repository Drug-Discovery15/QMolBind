"""Upper-confidence-bound acquisition for MINIMISATION targets: promising = low predicted energy OR high uncertainty."""
from __future__ import annotations

import numpy as np


def ucb_scores(mu: np.ndarray, sigma: np.ndarray, beta: float) -> np.ndarray:
    """Higher is better. In target (symlog) units: -mu + beta * sigma  (= -(mu - beta*sigma), the LCB of the energy)."""
    return -np.asarray(mu) + beta * np.asarray(sigma)


def select_top_b(mu: np.ndarray, sigma: np.ndarray, b: int, beta: float) -> np.ndarray:
    b = min(b, len(mu))
    return np.argsort(-ucb_scores(mu, sigma, beta), kind="stable")[:b]
