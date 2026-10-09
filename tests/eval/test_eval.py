import numpy as np
import pandas as pd
import pytest

from qumolbind.eval import metrics as M
from qumolbind.eval.stats import bootstrap_ci, bootstrap_diff_ci, mann_whitney


def test_report_module_imports() -> None:
    import qumolbind.eval.report as r  # noqa: F401  (regression: a syntax error here once broke `make smoke` at its last step)

    assert r.md_table(pd.DataFrame({"a": [1.5, np.nan], "b": ["x", "y"]})).count("|") > 4
    assert r.md_table(pd.DataFrame()) == "_no data_"


def test_bootstrap_ci_covers_truth_and_diff_sign() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 200)
    p, lo, hi = bootstrap_ci(x)
    assert lo < 5.0 < hi and lo <= p <= hi
    d, dlo, dhi = bootstrap_diff_ci(rng.normal(10, 1, 50), rng.normal(0, 1, 50))
    assert dlo > 0 and dlo < d < dhi


def test_mann_whitney_effect_size_direction() -> None:
    r = mann_whitney([5, 6, 7, 8], [1, 2, 3, 4])
    assert r["rank_biserial"] == pytest.approx(1.0) and r["p"] < 0.1
    assert mann_whitney([1, 2, 3, 4], [5, 6, 7, 8])["rank_biserial"] == pytest.approx(-1.0)


def _synthetic(vqc_better_on: list[str], targets=("A", "B"), n=12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for t in targets:
        for m in [*M.BASELINES, M.VQC]:
            for s in range(n):
                good = m == M.VQC and t in vqc_better_on
                e = rng.normal(100 if good else 500, 5)
                rows.append({"target": t, "method": m, "seed": s, "best_score": e, "success": bool(rng.random() < (0.95 if good else 0.4)),
                             "n_params": 1, "wall_s": 1.0, "best_rmsd": 1.0, "budget": 1})
    return pd.DataFrame(rows)


def _e7(mae: float) -> pd.DataFrame:
    return pd.DataFrame({"chi": [4, 4], "mae_mean_abs_diff": [mae, mae]})


def test_criterion_positive_branch_requires_two_targets_and_e7_clause() -> None:
    df = _synthetic(["A", "B"])
    ok = M.criterion(df, _e7(0.2))
    assert ok["n_target_wins"] == 2 and ok["branch"].startswith("VQC REPORTED")
    sim = M.criterion(df, _e7(0.01))  # MPS(chi=4) reproduces the output -> clause fails
    assert sim["branch"] == "NO DEMONSTRATED ADVANTAGE" and any("E7" in c for c in sim["failed_conditions"])
    one = M.criterion(_synthetic(["A"]), _e7(0.2))
    assert one["n_target_wins"] == 1 and one["branch"] == "NO DEMONSTRATED ADVANTAGE"
    none_e7 = M.criterion(df, None)
    assert none_e7["branch"] == "NO DEMONSTRATED ADVANTAGE" and any("E7 not run" in c for c in none_e7["failed_conditions"])


def test_criterion_no_advantage_when_equal() -> None:
    df = _synthetic([])
    out = M.criterion(df, _e7(0.3))
    assert out["n_target_wins"] == 0 and out["branch"] == "NO DEMONSTRATED ADVANTAGE"


def test_summary_table_columns() -> None:
    st = M.summary_table(_synthetic(["A"]))
    assert {"median_best_energy", "energy_ci_lo", "success_rate", "n_seeds"} <= set(st.columns)
    assert (st.energy_ci_lo <= st.median_best_energy).all() and (st.median_best_energy <= st.energy_ci_hi).all()
