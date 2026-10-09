import numpy as np
import openmm
import pytest

from qumolbind.active.multifidelity import MultiFidelityDataset, Record
from qumolbind.active.surrogate import FEAT_DIM, SurrogateEnsemble, from_target, pose_features, to_target
from qumolbind.active.ucb import select_top_b, ucb_scores

HAS_GPU = "OpenCL" in [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="Level-2 MD needs the OpenCL platform (CPU GBn2 is ~500 ms/eval)")


def test_target_transform_roundtrip() -> None:
    y = np.array([-500.0, -3.0, 0.0, 12.0, 4e3])
    assert np.allclose(from_target(to_target(y)), y, rtol=1e-9, atol=1e-9)


def _toy(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(0, 0.5, (n, FEAT_DIM)).astype(np.float32)
    return X, 200.0 * X[:, 0] - 80.0 * X[:, 3] - 100.0


def test_surrogate_learns_and_is_more_uncertain_off_distribution() -> None:
    rng = np.random.default_rng(0)
    X, y = _toy(60, rng)
    s = SurrogateEnsemble(5, epochs=300, seed=0).fit(X, y)
    mu, sd_in = s.predict_dg(X)
    assert np.corrcoef(mu, y)[0, 1] > 0.9
    far = X + 4.0
    _, sd_out = s.predict_dg(far)
    assert sd_out.mean() > 2 * sd_in.mean()


def test_untrained_surrogate_is_prior() -> None:
    mu, sd = SurrogateEnsemble().predict(np.zeros((3, FEAT_DIM)))
    assert np.all(mu == 0) and np.all(sd == 1)


def test_ucb_selection() -> None:
    mu = np.array([-2.0, 0.0, 1.0, -1.0])    # lower = better
    sd = np.array([0.1, 0.1, 3.0, 0.1])
    assert list(select_top_b(mu, sd, 1, beta=0.0)) == [0]        # pure exploitation
    assert list(select_top_b(mu, sd, 1, beta=5.0)) == [2]        # exploration dominates
    assert ucb_scores(mu, sd, 1.0)[0] == pytest.approx(2.1)
    assert len(select_top_b(mu, sd, 99, 1.0)) == 4


def test_dataset_fidelity_tags() -> None:
    d = MultiFidelityDataset()
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    d.add(Record(0, 0, f, l1_score=5.0))
    d.add(Record(0, 1, f, l1_score=6.0, selected_for_l2=True, l2_dg=-10.0))
    d.add(Record(0, 2, f, l1_score=7.0, selected_for_l2=True, l2_failed=True))
    assert len(d.labelled()) == 1
    X, y = d.xy()
    assert X.shape == (1, FEAT_DIM) and y[0] == -10.0
    df = d.to_frame()
    assert set(df.columns) >= {"fidelity_l1", "fidelity_l0", "fidelity_l2"}
    assert df.loc[1, "fidelity_l2"].startswith("L2") and df.loc[0, "fidelity_l2"] == ""


def test_pose_features_have_no_rmsd_inputs() -> None:
    f = pose_features(np.array([10.0, 20.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert f.shape == (FEAT_DIM,)


@needs_gpu
def test_slow_oracle_native_pose_finite_and_frozen_protein() -> None:
    from qumolbind.sim.oracle_slow import SlowOracle
    from qumolbind.sim.target import load_target

    t = load_target("3ert")
    so = SlowOracle(t.protein_pdb, t.ligand.mol, t.ligand.native, fast_oracle=t.make_oracle(), ligand_model=t.ligand)
    r = so.delta_g(t.ligand.native, md_ps=1.0, seed=0)
    assert np.isfinite(r.dg) and r.n_frames >= 1 and r.dg < 0
    assert t.rmsd(r.final_coords) < 4.0  # ligand stays in the pocket over 1 ps


@needs_gpu
def test_active_learning_loop_two_rounds(tmp_path) -> None:
    from qumolbind.active.loop import ActiveLearningLoop
    from qumolbind.baselines.common import make_problem

    prob = make_problem("3ert", {})
    al = {"rounds": 2, "M": 6, "b": 2, "beta": 1.0, "md_ps": 1.0, "ensemble": 3, "max_l1_score": 1e9, "min_sep_deg": 10.0}
    out = ActiveLearningLoop(prob, 200, al, {"n_qubits": 8, "n_layers": 2}, 1e-2, 0, tmp_path).run()
    assert len(out["rounds"]) == 2 and out["n_L2"] >= 1
    assert (tmp_path / "calibration.png").exists() and (tmp_path / "candidates.csv").exists()
    assert out["result"].calls_used <= 200
