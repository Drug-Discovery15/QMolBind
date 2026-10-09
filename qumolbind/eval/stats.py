"""Statistics: percentile bootstrap CIs (over seeds), Mann-Whitney U with rank-biserial effect size."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import mannwhitneyu

N_BOOT = 10_000


def bootstrap_ci(x, stat: Callable = np.median, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(point estimate, lo, hi): percentile bootstrap of ``stat`` over the entries of x (seeds)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    if stat is np.median:
        boots = np.median(x[idx], axis=1)
    elif stat is np.mean:
        boots = np.mean(x[idx], axis=1)
    else:
        boots = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_diff_ci(a, b, stat: Callable = np.median, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) for stat(a) - stat(b); the two groups' seeds are resampled independently."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ia, ib = rng.integers(0, len(a), (n_boot, len(a))), rng.integers(0, len(b), (n_boot, len(b)))
    if stat is np.median:
        d = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    elif stat is np.mean:
        d = np.mean(a[ia], axis=1) - np.mean(b[ib], axis=1)
    else:
        d = np.array([stat(a[i]) - stat(b[j]) for i, j in zip(ia, ib)])
    return float(stat(a) - stat(b)), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def mann_whitney(a, b) -> dict[str, float]:
    """Two-sided MWU of a vs b. rank_biserial = 2*U_a/(n_a n_b) - 1 in [-1, 1] (> 0: values of a tend to be larger)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 1 or len(b) < 1:
        return {"U": float("nan"), "p": float("nan"), "rank_biserial": float("nan")}
    r = mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(r.statistic), "p": float(r.pvalue), "rank_biserial": float(2 * r.statistic / (len(a) * len(b)) - 1)}
