"""Vectorised PoseEnv: in-process sync version and a multiprocessing version (one OpenMM context per worker)."""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np

from qumolbind.utils.seeding import derive_seed


class SyncVecEnv:
    """Sequential vector env with auto-reset. ``step`` returns the *post-reset* obs for finished envs and the
    terminal observation / infos under ``infos[i]['final_*']`` (CleanRL convention; time-limit truncation only)."""

    def __init__(self, env_fns: list) -> None:
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict]]:
        out = [e.reset(seed=None if seed is None else derive_seed(seed, i)) for i, e in enumerate(self.envs)]
        return np.stack([o for o, _ in out]), [i for _, i in out]

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, infos = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, te, tr, info = e.step(a)
            if te or tr:
                info["final_observation"] = o
                o, _ = e.reset()
            obs.append(o); rew.append(r); term.append(te); trunc.append(tr); infos.append(info)
        return np.stack(obs), np.array(rew, dtype=np.float32), np.array(term), np.array(trunc), infos

    @property
    def oracle_calls(self) -> int:
        return int(sum(e.oracle.calls for e in self.envs))

    def clip_rate(self) -> float:
        steps = sum(e.n_steps for e in self.envs)
        return sum(e.n_clipped for e in self.envs) / max(steps, 1)

    def close(self) -> None:
        pass


def _worker(conn, env_fn, seed: int) -> None:  # pragma: no cover - exercised via SubprocVecEnv test
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from qumolbind.utils.seeding import seed_everything

    seed_everything(seed)
    env = env_fn()
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send(env.reset(seed=data))
        elif cmd == "step":
            o, r, te, tr, info = env.step(data)
            if te or tr:
                info["final_observation"] = o
                o, _ = env.reset()
            conn.send((o, r, te, tr, info))
        elif cmd == "meta":
            conn.send((env.action_space, env.observation_space, env.oracle.calls, env.n_steps, env.n_clipped))
        elif cmd == "close":
            conn.close()
            break


class SubprocVecEnv:
    """One process (and OpenMM context) per env; ``env_fns`` must be picklable (top-level functions / partials)."""

    def __init__(self, env_fns: list, base_seed: int = 0) -> None:
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i, fn in enumerate(env_fns):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, fn, derive_seed(base_seed, i)), daemon=True)
            p.start()
            self.conns.append(parent); self.procs.append(p)
        self.n = len(env_fns)
        self.conns[0].send(("meta", None))
        self.single_action_space, self.single_observation_space = self.conns[0].recv()[:2]

    def reset(self, seed: int | None = None):
        for i, c in enumerate(self.conns):
            c.send(("reset", None if seed is None else derive_seed(seed, i)))
        out = [c.recv() for c in self.conns]
        return np.stack([o for o, _ in out]), [i for _, i in out]

    def step(self, actions: np.ndarray):
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        res = [c.recv() for c in self.conns]
        o, r, te, tr, info = zip(*res)
        return np.stack(o), np.array(r, dtype=np.float32), np.array(te), np.array(tr), list(info)

    def _meta(self) -> list[tuple[Any, ...]]:
        for c in self.conns:
            c.send(("meta", None))
        return [c.recv() for c in self.conns]

    @property
    def oracle_calls(self) -> int:
        return int(sum(m[2] for m in self._meta()))

    def clip_rate(self) -> float:
        m = self._meta()
        return sum(x[4] for x in m) / max(sum(x[3] for x in m), 1)

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
