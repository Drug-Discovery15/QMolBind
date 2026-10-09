"""E4: barren-plateau diagnostics. Gradient variance at random initialisation over >= 200 inits for n in {4,6,8,10}, L in {1,2,4,8}.

Cost C(theta) = mean_b <Z_0>(x_b) over real environment states x_b (random-policy rollouts on the target); the variance is taken over
inits of dC/dtheta, reported per parameter averaged over all parameters (and for one first-layer / one last-layer parameter).
Inits: 'near_identity' N(0, 0.1^2) (the training init) and 'uniform' U[0, 2pi) (the classic barren-plateau regime).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.quantum.torch_vqc import TorchVQC  # noqa: E402


def collect_states(target: str, n_states: int, seed: int = 0) -> np.ndarray:
    cache = ROOT / "results" / "E4" / f"states_{target}.npy"
    if cache.exists() and len(np.load(cache)) >= n_states:
        return np.load(cache)[:n_states]
    from qumolbind.baselines.common import make_problem
    from qumolbind.eval.policy_io import make_vec

    prob = make_problem(target, {})
    vec = make_vec(prob, 8)
    rng = np.random.default_rng(seed)
    obs, _ = vec.reset(seed=seed)
    states = [obs]
    while sum(len(s) for s in states) < n_states:
        obs, *_ = vec.step(rng.uniform(-1, 1, (8, vec.envs[0].K)))
        states.append(obs)
    arr = np.concatenate(states)[:n_states]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def grad_variance(n: int, L: int, init: str, states: torch.Tensor, n_inits: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vqc = TorchVQC(n, L, 1, init_std=0.1)
    grads = []
    for _ in range(n_inits):
        with torch.no_grad():
            vqc.theta.copy_(torch.as_tensor(rng.normal(0, 0.1, vqc.theta.shape) if init == "near_identity" else rng.uniform(0, 2 * np.pi, vqc.theta.shape)))
        vqc.zero_grad()
        vqc(states).mean().backward()
        grads.append(vqc.theta.grad.clone().reshape(-1).numpy())
    G = np.stack(grads)
    v = G.var(axis=0, ddof=1)
    return {"n_qubits": n, "n_layers": L, "init": init, "n_inits": n_inits, "n_params": G.shape[1], "var_mean_over_params": float(v.mean()),
            "var_median_over_params": float(np.median(v)), "var_first_layer_q0": float(v[0]), "var_mid_param": float(v[(L // 2) * n + n // 2]), "mean_abs_grad": float(np.abs(G).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--target", default=None)
    ap.add_argument("--inits", type=int, default=None)
    ap.add_argument("--ns", nargs="*", type=int, default=[4, 6, 8, 10])
    ap.add_argument("--layers", nargs="*", type=int, default=[1, 2, 4, 8])
    a = ap.parse_args()
    from qumolbind.utils.config import load_config, to_dict

    exp = to_dict(load_config([f"experiment={a.experiment}"]))["experiment"]
    target = a.target or exp["targets"][0]
    n_inits = a.inits or 200  # the brief requires >= 200 inits even for smoke
    states = torch.as_tensor(collect_states(target, 32), dtype=torch.float64)
    rows = []
    for init in ("near_identity", "uniform"):
        for n in a.ns:
            for L in a.layers:
                rows.append(grad_variance(n, L, init, states, n_inits, seed=1000 * n + L))
                print(rows[-1], flush=True)
    out = ROOT / "results" / f"E4_{exp['name']}.csv"
    pd.DataFrame(rows).assign(target_states=target).to_csv(out, index=False)
    print("wrote", out)
