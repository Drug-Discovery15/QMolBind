"""Result loading, per-method summaries and the mechanical evaluation of the pre-registered criterion (docs/PREREGISTRATION.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qumolbind.eval.stats import bootstrap_ci, bootstrap_diff_ci, mann_whitney

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ["random_search", "hill_climb", "cmaes", "ppo_mlp_matched", "ppo_mlp_large"]
VQC = "ppo_vqc"
E7_CHI, E7_TAU = 4, 0.05


def load_experiment(exp: str) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((ROOT / "results" / exp / "rows").glob("*.json"))]
    return pd.DataFrame(rows)


def load_curves(exp: str, target: str, method: str) -> np.ndarray:
    """[n_seeds, B] best-energy-so-far curves."""
    files = sorted((ROOT / "results" / exp / "curves").glob(f"{target}_{method}_*.npy"))
    return np.stack([np.load(f)[0] for f in files]) if files else np.zeros((0, 0))


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (tid, m), g in df.groupby(["target", "method"], sort=False):
        e = bootstrap_ci(g.best_score.to_numpy())
        s = bootstrap_ci(g.success.astype(float).to_numpy(), np.mean)
        out.append({"target": tid, "method": m, "n_seeds": len(g), "median_best_energy": e[0], "energy_ci_lo": e[1], "energy_ci_hi": e[2],
                    "success_rate": s[0], "success_ci_lo": s[1], "success_ci_hi": s[2], "n_params": int(g.n_params.median()),
                    "mean_wall_s": float(g.wall_s.mean()), "median_best_rmsd": float(g.best_rmsd.median())})
    return pd.DataFrame(out)


def criterion(df: pd.DataFrame, e7: pd.DataFrame | None, baselines=BASELINES, vqc: str = VQC) -> dict:
    """Mechanical verdict per docs/PREREGISTRATION.md: per-target details, failed conditions and the branch."""
    per_target, wins = [], 0
    for tid, g in df.groupby("target"):
        v = g[g.method == vqc]
        base = {m: g[g.method == m] for m in baselines if (g.method == m).any()}
        if v.empty or not base:
            per_target.append({"target": tid, "note": "missing VQC or baseline runs", "win": False})
            continue
        best_e = min(base, key=lambda m: base[m].best_score.median())
        best_s = max(base, key=lambda m: base[m].success.mean())
        d_e = bootstrap_diff_ci(base[best_e].best_score.to_numpy(), v.best_score.to_numpy())  # > 0 favours the VQC
        d_s = bootstrap_diff_ci(v.success.astype(float).to_numpy(), base[best_s].success.astype(float).to_numpy(), np.mean)  # > 0 favours the VQC
        win_e, win_s = d_e[1] > 0, d_s[1] > 0
        wins += int(win_e or win_s)
        per_target.append({
            "target": tid, "n_seeds_vqc": len(v), "best_baseline_energy": best_e, "energy_diff_baseline_minus_vqc": d_e[0],
            "energy_diff_ci": (d_e[1], d_e[2]), "energy_win": bool(win_e), "best_baseline_success": best_s,
            "success_diff_vqc_minus_baseline": d_s[0], "success_diff_ci": (d_s[1], d_s[2]), "success_win": bool(win_s), "win": bool(win_e or win_s),
            "mwu_energy_vqc_vs_best": mann_whitney(v.best_score, base[best_e].best_score),
        })
    failed = []
    if wins < 2:
        failed.append(f"VQC won on {wins} target(s); the criterion needs >= 2")
    mae = None
    e7_ok = None
    if e7 is None or e7.empty:
        failed.append("E7 not run (clause cannot be satisfied)")
    else:
        mae = float(e7[e7.chi == E7_CHI].mae_mean_abs_diff.mean())
        e7_ok = mae > E7_TAU
        if not e7_ok:
            failed.append(f"E7: MPS(chi={E7_CHI}) reproduces the VQC output (mean abs diff {mae:.4f} <= {E7_TAU})")
    return {"per_target": per_target, "n_target_wins": wins, "e7_mae_chi4": mae, "e7_clause_holds": e7_ok, "failed_conditions": failed,
            "branch": "VQC REPORTED AS BETTER THAN CLASSICAL BASELINES" if not failed else "NO DEMONSTRATED ADVANTAGE"}
