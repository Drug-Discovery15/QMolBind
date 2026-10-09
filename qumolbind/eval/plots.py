"""Figures (matplotlib, Agg). Curves are median +- IQR across seeds; energies use a symmetric-log axis (clash energies span 10 decades)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#ad494a"]


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_curves(curves: dict[str, np.ndarray], title: str, out: Path, ylabel: str = "best energy so far (kJ/mol, symlog axis)") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (m, c) in enumerate(curves.items()):
        if c.size == 0:
            continue
        x = np.arange(1, c.shape[1] + 1)
        med, lo, hi = np.median(c, 0), np.quantile(c, 0.25, 0), np.quantile(c, 0.75, 0)
        col = PALETTE[i % len(PALETTE)]
        ax.plot(x, med, color=col, lw=1.6, label=f"{m} (n={c.shape[0]})")
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("oracle calls (Level-1 evaluations)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    _save(fig, out)


def plot_barren(df, out: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, init in zip(axs, ("near_identity", "uniform")):
        d = df[df.init == init]
        for i, L in enumerate(sorted(d.n_layers.unique())):
            g = d[d.n_layers == L].sort_values("n_qubits")
            ax.plot(g.n_qubits, g.var_mean_over_params, "o-", color=PALETTE[i], label=f"L={L}")
        ax.set_yscale("log")
        ax.set_xlabel("qubits n")
        ax.set_title(f"init: {init}", fontsize=10)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("Var[dC/dtheta] (mean over parameters)")
    axs[0].legend(fontsize=8)
    _save(fig, out)


def plot_lines(x, ys: dict[str, tuple[np.ndarray, np.ndarray]], xlabel: str, ylabel: str, title: str, out: Path, logx: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (k, (m, s)) in enumerate(ys.items()):
        ax.errorbar(x, m, yerr=s, fmt="o-", color=PALETTE[i % len(PALETTE)], capsize=3, label=k)
    if logx:
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    if len(ys) > 1:
        ax.legend(fontsize=8)
    _save(fig, out)
