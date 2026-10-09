"""E7: classical simulability. For each trained ppo_vqc policy: the policy's visited states are fed through the circuit
(a) exactly (torch statevector) and (b) with Qiskit-Aer MPS at bond dimension chi in {2,4,8,16}.
Reported: mean |<Z>_MPS - <Z>_exact| (the pre-registered E7 clause uses chi=4), plus output-state entanglement entropy
(trained vs. fresh init, and from the training logs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.eval.policy_io import load_vqc_actor, rollout  # noqa: E402
from qumolbind.quantum.encoding import amplitude_state  # noqa: E402
from qumolbind.quantum.mps import mps_expectations  # noqa: E402
from qumolbind.quantum.runner import make_vqc_actor  # noqa: E402
from qumolbind.quantum.torch_vqc import entanglement_entropy  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402

CHIS = [2, 4, 8, 16]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--n-states", type=int, default=48)
    a = ap.parse_args()
    d = to_dict(load_config([f"experiment={a.experiment}"]))
    exp = d["experiment"]
    rows, ent_rows = [], []
    for tid in exp["targets"]:
        prob = make_problem(tid, d["env"])
        for s in exp["seeds"]:
            ck = ROOT / "results" / exp["name"] / "ckpt" / f"{tid}_ppo_vqc_{s}.pt"
            if not ck.exists():
                print("missing", ck)
                continue
            actor = load_vqc_actor(ck, prob, d["actor"])
            vqc, n = actor.vqc, actor.vqc.n
            states, *_ = rollout(prob, lambda o: actor(o), actor.log_std.detach(), 8, seed=20_000 + s)
            states = states[np.random.default_rng(s).permutation(len(states))[: a.n_states]]
            x = torch.as_tensor(states, dtype=torch.float64)
            with torch.no_grad():
                exact = vqc(x).numpy()
                psi, _ = amplitude_state(x, n)
            th = vqc.theta.detach().numpy()
            for chi in CHIS:
                mps = np.stack([mps_expectations(th, psi[i].numpy(), n, vqc.K, chi, vqc.rotations, vqc.entangler) for i in range(len(x))])
                rows.append({"target": tid, "seed": s, "chi": chi, "n_states": len(x), "mae_mean_abs_diff": float(np.abs(mps - exact).mean()),
                             "max_abs_diff": float(np.abs(mps - exact).max())})
                print(rows[-1], flush=True)
            with torch.no_grad():
                fresh = make_vqc_actor(256, vqc.K, {k: v for k, v in d["actor"].items() if k != "lr"})
                for name, m in (("trained", vqc), ("fresh_init", fresh.vqc)):
                    st = m.final_state(x)
                    ents = [entanglement_entropy(st, n, c).mean().item() for c in range(1, n)]
                    ent_rows.append({"target": tid, "seed": s, "which": name, "entropy_mid_cut_bits": ents[n // 2 - 1], "entropy_max_cut_bits": max(ents),
                                     "max_possible_mid_cut_bits": n // 2})
            log = ROOT / "results" / exp["name"] / "logs" / f"{tid}_ppo_vqc_{s}" / "metrics.csv"
            if log.exists():
                lg = pd.read_csv(log)
                if "entanglement_entropy" in lg:
                    ent_rows.append({"target": tid, "seed": s, "which": "training_log_first_update", "entropy_mid_cut_bits": float(lg.entanglement_entropy.iloc[0])})
                    ent_rows.append({"target": tid, "seed": s, "which": "training_log_last_update", "entropy_mid_cut_bits": float(lg.entanglement_entropy.iloc[-1])})
    pd.DataFrame(rows).to_csv(ROOT / "results" / f"E7_{exp['name']}.csv", index=False)
    pd.DataFrame(ent_rows).to_csv(ROOT / "results" / f"E7_entropy_{exp['name']}.csv", index=False)
    print("wrote E7 results")
