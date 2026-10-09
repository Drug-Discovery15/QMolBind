import numpy as np
import pytest

from qumolbind.baselines import REGISTRY
from qumolbind.baselines.common import BudgetExhausted, make_problem


@pytest.fixture(scope="module")
def problem():
    return make_problem("1cil", {"episode_len": 5})


@pytest.mark.parametrize("name", list(REGISTRY))
def test_budget_respected_and_curve_monotone(problem, name) -> None:
    res = REGISTRY[name](problem, 120, 0)
    assert res.calls_used == 120 and res.budget == 120
    c = res.curve
    assert np.all(np.isfinite(c)) and np.all(np.diff(c) <= 1e-12)
    assert res.best_score == pytest.approx(c[-1])
    assert 0.0 <= res.best_rmsd < 20.0


def test_same_seed_same_result(problem) -> None:
    a = REGISTRY["random_search"](problem, 40, 3)
    b = REGISTRY["random_search"](problem, 40, 3)
    assert a.best_score == b.best_score


def test_tracker_raises_when_exhausted(problem) -> None:
    tr = problem.tracker(2)
    c = problem.target.ligand.native
    tr.evaluate(c); tr.evaluate(c)
    with pytest.raises(BudgetExhausted):
        tr.evaluate(c)


class _FakeLigand:
    K = 6
    native = np.zeros((1, 3))

    def __init__(self) -> None:
        self.xstar = np.linspace(-120, 150, self.K)

    def set_torsions_deg(self, coords, t):
        return np.asarray(t, dtype=float).reshape(1, -1)


class _Terms:
    def __init__(self, s):
        self.score = s


class _FakeOracle:
    def __init__(self, xstar):
        self.xstar = xstar

    def evaluate(self, c):
        d = (c.ravel() - self.xstar + 180) % 360 - 180
        return _Terms(float(np.sum(d**2)) / 100.0)


def test_search_methods_work_on_smooth_periodic_objective() -> None:
    """Validates the implementations independent of the (rugged) docking landscape."""
    from types import SimpleNamespace

    from qumolbind.baselines import cmaes, hill_climb, random_search
    from qumolbind.baselines.common import Problem

    lig = _FakeLigand()
    tgt = SimpleNamespace(ligand=lig, rmsd=lambda c: 0.0)
    prob = Problem("fake", tgt, _FakeOracle(lig.xstar), {})
    res = {m.__name__.split(".")[-1]: m.run(prob, 600, 0) for m in (random_search, hill_climb, cmaes)}
    assert res["cmaes"].best_score < 1e-2
    assert res["hill_climb"].best_score < res["random_search"].best_score
    assert res["cmaes"].best_score < res["random_search"].best_score
