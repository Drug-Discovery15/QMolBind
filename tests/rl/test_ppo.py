import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from qumolbind.env.vec import SyncVecEnv
from qumolbind.rl.actors import MLPActor, matched_mlp_hidden, mlp_param_count
from qumolbind.rl.critic import Critic
from qumolbind.rl.ppo import PPO, PPOConfig
from qumolbind.utils.seeding import seed_everything


class QuadEnv(gym.Env):
    """Trivial task: drive x in R^2 to the origin. reward = -(|x'|^2 - |x|^2)."""

    def __init__(self, T: int = 10) -> None:
        self.T = T
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        self.n_oracle = self.n_steps = self.n_clipped = 0

    def _obs(self):
        return np.concatenate([self.x, [self.t / self.T, 0.0]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.np_random.uniform(-2, 2, 2)
        self.t = 0
        return self._obs(), {}

    def step(self, a):
        old = float(self.x @ self.x)
        self.x = self.x + 0.5 * np.clip(a, -1, 1)
        self.t += 1
        self.n_steps += 1
        return self._obs(), -(float(self.x @ self.x) - old), False, self.t >= self.T, {"energy": float(self.x @ self.x)}


def _final_energy(actor: MLPActor, n: int = 200) -> float:
    env, out = QuadEnv(), []
    for i in range(n):
        o, _ = env.reset(seed=1000 + i)
        done = False
        while not done:
            with torch.no_grad():
                a = actor(torch.as_tensor(o)[None])[0].numpy()
            o, _, _, done, info = env.step(a)
        out.append(info["energy"])
    return float(np.mean(out))


def test_ppo_learns_quadratic() -> None:
    seed_everything(0)
    vec = SyncVecEnv([QuadEnv for _ in range(8)])
    actor, critic = MLPActor(4, 2, (32, 32)), Critic(4, (32, 32))
    before = _final_energy(actor)
    cfg = PPOConfig(n_envs=8, rollout_steps=10, epochs=6, minibatches=2, actor_lr=3e-3, critic_lr=3e-3, ent_coef=0.0)
    PPO(vec, actor, critic, cfg).train(max_updates=150)
    after = _final_energy(actor)
    assert after < 0.25 * before, (before, after)


def test_matched_mlp_sizing() -> None:
    h = matched_mlp_hidden(256, 8, 65_792 + 0)  # VQC with trainable 256->256 projection
    assert abs(mlp_param_count(256, h, 8) - 65_792) / 65_792 < 0.01
    h1 = matched_mlp_hidden(256, 8, 32)  # infeasible target -> smallest network
    assert h1 == (1,)
