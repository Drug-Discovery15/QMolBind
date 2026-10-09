"""QPPO: the shared PPO code with the VQC actor (and the callback that logs quantum diagnostics)."""
from __future__ import annotations

import numpy as np
import torch

from qumolbind.baselines.common import Problem, RunResult
from qumolbind.baselines.ppo_mlp import run_ppo
from qumolbind.rl.actors import VQCActor

ACTOR_KEYS = {"n_qubits", "n_layers", "rotations", "entangler", "encoding", "init_std", "trainable_scale", "input_projection", "shots"}


def make_vqc_actor(obs_dim: int, K: int, actor_cfg: dict | None = None) -> VQCActor:
    kw = {k: v for k, v in (actor_cfg or {}).items() if k in ACTOR_KEYS}
    return VQCActor(obs_dim, K, **kw)


def quantum_diagnostics(ppo, row: dict) -> None:
    """Entanglement entropy of the output state (middle cut), mean input norm (discarded by amplitude encoding)."""
    actor = ppo.actor
    if not hasattr(actor, "vqc") or not hasattr(ppo, "last_batch_obs"):
        return
    with torch.no_grad():
        obs = ppo.last_batch_obs[:64].to(torch.float64)
        row["entanglement_entropy"] = float(actor.vqc.entanglement_entropy(obs).mean())
        row["input_norm"] = float(actor.vqc.input_norm(obs).mean())
        row["mean_abs_action_mean"] = float(actor(obs.to(torch.float32)).abs().mean())


def run_ppo_vqc(problem: Problem, budget: int, seed: int, lr: float = 1e-2, actor_cfg: dict | None = None,
                ppo_kw: dict | None = None, logger=None, method: str = "ppo_vqc", return_ppo: bool = False,
                ckpt_path: str | None = None, init_state: dict | None = None, log_dir: str | None = None, **hp):
    cfg = dict(actor_cfg or {})
    cfg.pop("lr", None)  # lr is a separate argument (swept {3e-3, 1e-2, 3e-2})
    def factory(d: int, k: int) -> VQCActor:
        return make_vqc_actor(d, k, cfg)

    res, ppo = run_ppo(problem, budget, seed, method, factory, lr, ppo_kw, logger,
                       init_state=init_state, ckpt_path=ckpt_path, on_update=quantum_diagnostics, log_dir=log_dir, hparams={"n_quantum_params": factory(problem.env_cfg.get("state_dim", 256), problem.target.ligand.K).n_quantum_params(),
                        "n_classical_mean_params": factory(problem.env_cfg.get("state_dim", 256), problem.target.ligand.K).n_classical_mean_params(),
                        **{f"vqc_{k}": v for k, v in cfg.items() if k in ACTOR_KEYS}})
    return (res, ppo) if return_ppo else res


# Named VQC variants for E1-E3/E5 (overrides applied on top of the base actor config).
VARIANTS: dict[str, dict] = {
    "ppo_vqc": {},                                                  # (i) fixed zero-padding, no projection, 32 params
    "ppo_vqc_proj": {"input_projection": True},                     # (ii) trainable classical projection (counted)
    "ppo_vqc_noent": {"entangler": "none"},                         # E2: no entanglement
    "ppo_vqc_cz": {"entangler": "cz"},
    "ppo_vqc_ryrz": {"rotations": "ryrz"},                          # E3
    "ppo_vqc_angle": {"encoding": "angle", "input_projection": True},  # E3: angle encoding needs d->n projection (counted)
    "ppo_vqc_shots4096": {"shots": 4096}, "ppo_vqc_shots1024": {"shots": 1024}, "ppo_vqc_shots256": {"shots": 256},  # E5
}


def make_variant_runner(name: str):
    over = VARIANTS[name]

    def run(problem: Problem, budget: int, seed: int, lr: float = 1e-2, actor_cfg: dict | None = None, **kw) -> RunResult:
        cfg = {**(actor_cfg or {}), **over}
        return run_ppo_vqc(problem, budget, seed, lr=lr, actor_cfg=cfg, method=name, **{k: v for k, v in kw.items() if k != "actor_cfg"})

    return run
