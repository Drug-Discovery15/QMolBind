"""Stage 9: inference-only check of a TRAINED VQC on an IBM backend (<= 20 states). Hardware is OFF by default.

  python scripts/run_hardware_eval.py                       # dry run (no network): FakeFez noise-model simulation
  python scripts/run_hardware_eval.py --allow-hardware      # needs IBM_QUANTUM_TOKEN; otherwise exits cleanly with instructions

No training happens here, no proprietary data is used (states come from the public-PDB pilot environment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSTRUCTIONS = """Hardware run NOT executed.
To run on real hardware you must (1) pass --allow-hardware and (2) export IBM_QUANTUM_TOKEN=<your IBM Quantum Platform API key>
(optionally IBM_QUANTUM_INSTANCE=<instance CRN>) and (3) accept that jobs consume QPU time. Nothing is ever sent without both."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-hardware", action="store_true")
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--ckpt", default=None, help="trained ppo_vqc checkpoint (default: first one under results/<experiment>/ckpt)")
    ap.add_argument("--target", default=None)
    ap.add_argument("--n-states", type=int, default=10)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    if a.n_states > 20:
        print("refusing: at most 20 states")
        return 2

    from qumolbind.baselines.common import make_problem
    from qumolbind.eval.policy_io import load_vqc_actor, rollout
    from qumolbind.quantum.encoding import amplitude_state
    from qumolbind.quantum.hardware import run_dry, run_hardware
    from qumolbind.utils.config import load_config, to_dict

    d = to_dict(load_config([f"experiment={a.experiment}"]))
    ck = Path(a.ckpt) if a.ckpt else next(iter(sorted((ROOT / "results" / a.experiment / "ckpt").glob("*_ppo_vqc_*.pt"))), None)
    if ck is None or not ck.exists():
        print("no trained ppo_vqc checkpoint found; run `python scripts/run_experiments.py --methods ppo_vqc` first")
        return 1
    tid = a.target or torch.load(ck, weights_only=False)["target"]
    prob = make_problem(tid, d["env"])
    actor = load_vqc_actor(ck, prob, d["actor"])
    states, *_ = rollout(prob, lambda o: actor(o), actor.log_std.detach(), 4, seed=777)
    states = states[np.random.default_rng(0).permutation(len(states))[: a.n_states]]
    psi, _ = amplitude_state(torch.as_tensor(states, dtype=torch.float64), actor.vqc.n)
    theta, n, K = actor.vqc.theta.detach().numpy(), actor.vqc.n, actor.vqc.K
    args = dict(theta=theta, states=psi.numpy(), n=n, K=K, shots=a.shots, rotations=actor.vqc.rotations, entangler=actor.vqc.entangler)

    if a.allow_hardware and os.environ.get("IBM_QUANTUM_TOKEN"):
        rep = run_hardware(**args, backend_name=a.backend, allow_hardware=True)
    else:
        if a.allow_hardware:
            print(INSTRUCTIONS)
            print("Falling back to the offline dry run:\n")
        rep = run_dry(**args)
    out = ROOT / "results" / f"hardware_eval_{a.experiment}.json"
    out.write_text(json.dumps(rep.__dict__, indent=2))
    print(f"[{rep.kind}] backend={rep.backend} shots={rep.shots} states={rep.n_states} "
          f"mean|err|={rep.mean_abs_error:.4f} max|err|={rep.max_abs_error:.4f} 2q-gates/circuit={rep.two_qubit_gates_mean:.0f} depth={rep.depth_mean:.0f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
