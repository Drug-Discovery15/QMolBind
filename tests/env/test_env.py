import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from qumolbind.env.pose_env import make_env
from qumolbind.env.state import OFF_LIG, build_state
from qumolbind.env.vec import SyncVecEnv


@pytest.fixture(scope="module")
def env():
    return make_env("1cil", {"episode_len": 5})


def test_env_checker(env) -> None:
    check_env(env, skip_render_check=True)


def test_episode_and_info(env) -> None:
    obs, info = env.reset(seed=0)
    assert obs.shape == (256,) and obs.dtype == np.float32
    n0 = env.oracle.calls
    for t in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert {"energy", "rmsd_native", "oracle_calls"} <= set(info)
        assert np.isfinite(r) and abs(r) <= env.reward_clip
    assert trunc and not term
    assert env.oracle.calls == n0 + 5


def test_no_label_leakage(env) -> None:
    """State must not depend on RMSD / native coordinates: identical torsions+terms -> identical state."""
    env.reset(seed=3)
    s1 = env._obs()
    env.target.rmsd.native_heavy = env.target.rmsd.native_heavy + 5.0  # corrupt the evaluation-only reference
    s2 = env._obs()
    assert np.array_equal(s1, s2)


def test_protein_unchanged_by_steps(env) -> None:
    before = env.oracle.protein_positions_nm.copy()
    env.reset(seed=1)
    for _ in range(5):
        env.step(env.action_space.sample())
    assert np.array_equal(before, env.oracle.protein_positions_nm)


def test_state_layout_zero_padded() -> None:
    s = build_state(np.array([10.0, 20.0]), np.zeros(3), 1, 20)
    assert s.shape == (256,) and np.all(s[OFF_LIG:] == 0)


def test_sync_vec_autoreset() -> None:
    fns = [lambda: make_env("1cil", {"episode_len": 2}) for _ in range(2)]
    v = SyncVecEnv(fns)
    obs, _ = v.reset(seed=0)
    for _ in range(3):
        obs, r, te, tr, infos = v.step(np.zeros((2, 3)))
    assert obs.shape == (2, 256) and v.oracle_calls >= 8


def _fn():
    return make_env("1cil", {"episode_len": 2})


def test_subproc_vec_env() -> None:
    from qumolbind.env.vec import SubprocVecEnv

    v = SubprocVecEnv([_fn, _fn], base_seed=0)
    try:
        obs, _ = v.reset(seed=0)
        for _ in range(3):
            obs, r, te, tr, infos = v.step(np.zeros((2, 3)))
        assert obs.shape == (2, 256) and v.oracle_calls >= 8
        assert not np.array_equal(obs[0], obs[1])  # workers are seeded differently
    finally:
        v.close()


def test_rollout_helper_depends_on_policy_and_keeps_episode_end_stats() -> None:
    """Regression: evaluating through an auto-resetting vec env returned the NEXT episode's start for every policy."""
    import torch

    from qumolbind.baselines.common import make_problem
    from qumolbind.eval.policy_io import rollout

    prob = make_problem("1cil", {"episode_len": 4})
    ls = torch.full((3,), -3.0)
    _, b0, _, e0, f0 = rollout(prob, lambda o: torch.zeros(o.shape[0], 3), ls, 3, seed=1, sample=False)
    _, b1, _, e1, f1 = rollout(prob, lambda o: torch.ones(o.shape[0], 3), ls, 3, seed=1, sample=False)
    assert np.array_equal(e0, e1)                 # identical starts
    assert not np.allclose(f0, f1)                # different policies -> different final energies
    assert np.allclose(f0, e0)                    # zero action never moves the pose
    assert np.all(b0 <= e0 + 1e-9)


def test_elite_restart_starts_from_archive_and_defaults_off() -> None:
    from qumolbind.baselines.common import make_problem
    from qumolbind.baselines.ppo_mlp import build_vec_env

    prob = make_problem("1cil", {"episode_len": 3})
    tr = prob.tracker(400)
    rng = np.random.default_rng(0)
    for _ in range(30):
        tr.evaluate(prob.target.ligand.randomize(rng))
    assert 3 <= len(tr.archive) <= tr.elite_k and tr.archive[0][0] == tr.best_score
    assert [a[0] for a in tr.archive] == sorted(a[0] for a in tr.archive)
    from qumolbind.env.pose_env import make_env

    env = make_env("1cil", {"episode_len": 3, "elite_start_prob": 1.0}, target=prob.target, oracle=prob.oracle, wrap_oracle=lambda o: tr)
    env.reset(seed=1)
    assert any(np.allclose(env.coords, c) for _, c in tr.archive)        # started from an elite pose
    env0 = make_env("1cil", {"episode_len": 3}, target=prob.target, oracle=prob.oracle, wrap_oracle=lambda o: tr)
    assert env0.elite_start_prob == 0.0


def test_greedy_accept_never_worsens_current_pose_and_reward_reflects_proposal() -> None:
    env = make_env("1cil", {"episode_len": 30, "greedy_accept": True})
    env.reset(seed=2)
    cur = env.last_terms.score
    saw_reject = False
    rng = np.random.default_rng(0)
    for _ in range(30):
        c0 = env.coords.copy()
        _, r, _, _, info = env.step(rng.uniform(-1, 1, env.K))
        assert env.last_terms.score <= cur + 1e-9           # accepted pose never gets worse
        if env.last_terms.score == cur:
            saw_reject = True
            assert np.allclose(env.coords, c0)              # rejected -> pose unchanged
        cur = env.last_terms.score
    assert saw_reject and np.isfinite(info["best_energy"])
    assert info["best_energy"] <= cur + 1e-9


def test_zero_mean_actor_learns_only_log_std() -> None:
    import torch

    from qumolbind.rl.actors import ZeroMeanActor

    a = ZeroMeanActor(3, -1.0)
    assert torch.all(a(torch.randn(5, 256)) == 0) and a.n_mean_params() == 1 and a.log_std.requires_grad
