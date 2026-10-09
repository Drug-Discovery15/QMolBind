import numpy as np
import pytest

from qumolbind.sim.target import load_target


@pytest.fixture(scope="module")
def target():
    return load_target("1cil")


@pytest.fixture(scope="module")
def oracle(target):
    return target.make_oracle(precision="double")


def test_decomposition_matches_full_openmm(target, oracle) -> None:
    rng = np.random.default_rng(0)
    for coords in [target.ligand.native, target.ligand.randomize(rng)]:
        e = oracle.evaluate(coords)
        assert e.e_int == pytest.approx(oracle.reference_interaction(coords), abs=0.05)
        assert e.e_int == pytest.approx(e.vdw + e.elec + e.solv)


def test_calls_counted(target) -> None:
    o = target.make_oracle()
    for _ in range(3):
        o.evaluate(target.ligand.native)
    assert o.calls == 3


def test_native_beats_median_random(target, oracle) -> None:
    rng = np.random.default_rng(1)
    rand = [oracle.evaluate(target.ligand.randomize(rng)).score for _ in range(200)]
    assert oracle.evaluate(target.ligand.native).score < np.median(rand)


def test_360_rotation_invariance(target, oracle) -> None:
    lm = target.ligand
    base = lm.randomize(np.random.default_rng(2))
    e0 = oracle.evaluate(base).score
    for k in range(lm.K):
        d = np.zeros(lm.K)
        d[k] = 360.0
        assert oracle.evaluate(lm.apply_torsion_deltas(base, d)).score == pytest.approx(e0, abs=1e-3)


def test_protein_frozen_even_with_minimisation(target) -> None:
    o = target.make_oracle(minimize_iters=15, precision="double")
    o.evaluate(target.ligand.native)
    pos = o.ctx_c.getState(getPositions=True).getPositions(asNumpy=True)._value
    assert np.abs(np.array(pos[: o.n_prot]) - o.protein_positions_nm).max() < 1e-9
    assert not np.allclose(np.array(pos[o.n_prot :]), target.ligand.native / 10.0)  # ligand did relax


def test_ligand_parametrization_charges_sum(oracle, target) -> None:
    assert oracle.params.charges.sum() == pytest.approx(0.0, abs=1e-9)


def test_nonfinite_energy_is_capped_and_counted(target) -> None:
    from qumolbind.sim.oracle_fast import NONFINITE_CAP

    o = target.make_oracle()
    bad = target.ligand.native.copy()
    bad[0] = np.nan                      # NaN coordinate -> NaN energy
    e = o.evaluate(bad)
    assert np.isfinite(e.score) and e.score == NONFINITE_CAP and o.n_nonfinite == 1
    from qumolbind.rl.budget import Tracker

    t = Tracker(o, 5, target.rmsd)
    t.evaluate(target.ligand.native)
    assert t.n_nonfinite == 0
