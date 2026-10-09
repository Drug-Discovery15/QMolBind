"""Phase B of docs/PREREGISTRATION_V2.md: confirmation on fresh seeds, mechanical V2 verdict -> REPORT_V2.md.

Reads results/explore_selection.json (written by explore_vqc.py), runs seeds 200..209 on all targets at B=2000, evaluates the V2 criterion
(with the E7 MPS clause on the subject design) and the secondary 'parity with PPO-MLP' statement. Resumable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines import REGISTRY  # noqa: E402
from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.eval import metrics as M  # noqa: E402
from qumolbind.eval import plots  # noqa: E402
from qumolbind.eval.policy_io import load_vqc_actor, rollout  # noqa: E402
from qumolbind.eval.report import md_table  # noqa: E402
from qumolbind.eval.stats import bootstrap_diff_ci  # noqa: E402
from qumolbind.quantum.mps import mps_expectations, mps_expectations_reupload  # noqa: E402
from qumolbind.quantum.encoding import amplitude_state  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402
from scripts.explore_vqc import registry as ppo_registry, run_call  # noqa: E402

SEEDS = list(range(200, 210))
TARGETS = ["3ert", "1uyd", "1eve"]
B = 2000
SEARCH = ["random_search", "hill_climb", "cmaes"]
PPO_BASE = ["ppo_mlp_matched", "ppo_mlp_large", "ppo_mlp_matched_reup"]
BASELINES = [*SEARCH, *PPO_BASE]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    a = ap.parse_args()
    sel = json.loads((ROOT / "results" / "explore_selection.json").read_text())
    subject = sel["subject"]
    tuning = json.loads((ROOT / "results" / "tuning_pilot.json").read_text())
    d = to_dict(load_config(["experiment=pilot"]))
    actor_cfg = {k: v for k, v in d["actor"].items() if k != "lr"}
    rows_dir = ROOT / "results" / "confirm" / "rows"
    ck_dir = ROOT / "results" / "confirm" / "ckpt"
    rows_dir.mkdir(parents=True, exist_ok=True); ck_dir.mkdir(parents=True, exist_ok=True)
    preg = ppo_registry()
    methods = [*BASELINES, "ppo_vqc", *([subject] if subject != "ppo_vqc" else [])]
    for tid in TARGETS:
        prob = make_problem(tid, d["env"])
        print(f"== confirm target {tid}", flush=True)
        for m in methods:
            for s in a.seeds:
                f = rows_dir / f"{tid}_{m}_{s}.json"
                if f.exists():
                    continue
                ck = ck_dir / f"{tid}_{m}_{s}.pt"
                if m in SEARCH:
                    hp = tuning.get(f"{tid}/{m}", {}).get("hparams", {})
                    r = REGISTRY[m](prob, B, s, **hp)
                else:
                    cfg = sel["configs"][m]
                    kw = {"actor_cfg": actor_cfg} if m != "ppo_mlp_large" else {}
                    if m in ("ppo_vqc", subject):
                        kw["ckpt_path"] = str(ck)
                    r = preg[m](prob, B, s, lr=cfg["lr"], ppo_kw=dict(cfg["ppo_kw"]), **kw)
                (ROOT / "results" / "confirm" / "curves").mkdir(parents=True, exist_ok=True)
                np.save(ROOT / "results" / "confirm" / "curves" / f"{tid}_{m}_{s}.npy", np.stack([r.curve, r.rmsd_curve]))
                row = r.row(); row["method"] = m
                f.write_text(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}))
                print(f"  {m:22s} seed={s} best={r.best_score:10.1f} rmsd={r.best_rmsd:4.2f} params={r.n_params} ({r.wall_s:.0f}s)", flush=True)

    df = pd.DataFrame([json.loads(p.read_text()) for p in sorted(rows_dir.glob("*.json"))])

    # ---- E7 clause on the subject (chi = 4 MPS vs exact on visited states)
    e7_rows = []
    for tid in TARGETS:
        prob = make_problem(tid, d["env"])
        for s in a.seeds:
            ck = ck_dir / f"{tid}_{subject}_{s}.pt"
            if not ck.exists():
                continue
            actor = load_vqc_actor(ck, prob, d["actor"], method=subject)
            states, *_ = rollout(prob, lambda o: actor(o), actor.log_std.detach(), 6, seed=30_000 + s)
            states = states[np.random.default_rng(s).permutation(len(states))[:24]]
            x = torch.as_tensor(states, dtype=torch.float64)
            with torch.no_grad():
                exact = actor.vqc(x).numpy()
            v = actor.vqc
            for chi in (2, 4, 8, 16):
                if v.encoding == "reupload":
                    z = np.stack([mps_expectations_reupload(v, xi.numpy(), chi) for xi in x])
                else:
                    psi, _ = amplitude_state(x, v.n)
                    z = np.stack([mps_expectations(v.theta.detach().numpy(), psi[i].numpy(), v.n, v.K, chi, v.rotations, v.entangler) for i in range(len(x))])
                    if v.r_scale is not None:
                        z = z * v.r_scale.detach().numpy() + v.r_bias.detach().numpy()
                e7_rows.append({"target": tid, "seed": s, "chi": chi, "mae_mean_abs_diff": float(np.abs(z - exact).mean())})
    e7 = pd.DataFrame(e7_rows)
    e7.to_csv(ROOT / "results" / "confirm_E7.csv", index=False)
    verdict = M.criterion(df, e7 if len(e7) else None, baselines=BASELINES, vqc=subject)

    # ---- secondary: parity with the best PPO-MLP
    parity = []
    for tid in TARGETS:
        g = df[df.target == tid]
        mlp = {m: g[g.method == m] for m in PPO_BASE if (g.method == m).any()}
        if not mlp or not (g.method == subject).any():
            continue
        bm = min(mlp, key=lambda m: mlp[m].best_score.median())
        dd = bootstrap_diff_ci(mlp[bm].best_score.to_numpy(), g[g.method == subject].best_score.to_numpy())
        parity.append({"target": tid, "best PPO-MLP": bm, "median E diff (MLP - subject) [95% CI]": f"{dd[0]:.4g} [{dd[1]:.4g}, {dd[2]:.4g}]",
                       "CI contains 0 (parity)": bool(dd[1] <= 0 <= dd[2]), "subject better (CI > 0)": bool(dd[1] > 0), "subject worse (CI < 0)": bool(dd[2] < 0)})
    par_ok = bool(parity) and all(p["CI contains 0 (parity)"] or p["subject better (CI > 0)"] for p in parity)

    # ---- report
    L = ["# QuMolBind V2 report - exploratory VQC redesign, confirmed on fresh seeds", "",
         "_Generated by `scripts/confirm_vqc.py`; protocol fixed in `docs/PREREGISTRATION_V2.md` before any V2 run._", "",
         f"- subject design (selected in the exploration phase by the pre-registered rule): **`{subject}`**; exploration scores in `results/explore_selection.json`",
         f"- confirmation: seeds {min(a.seeds)}-{max(a.seeds)} ({len(a.seeds)}), targets {', '.join(TARGETS)}, B = {B} oracle calls per run",
         f"- exploration configs chosen: " + ", ".join(f"`{m}`={sel['configs'][m]['lr']:g}/{sel['configs'][m]['setting']}" for m in preg), "",
         "## V2 verdict (mechanical)", "", f"**`{verdict['branch']}`**", ""]
    L += ["Conditions not met:", *[f"- {c}" for c in verdict["failed_conditions"]], ""] if verdict["failed_conditions"] else []
    rows = []
    for t in verdict["per_target"]:
        if "note" in t:
            continue
        rows.append({"target": t["target"], "best baseline (energy)": t["best_baseline_energy"],
                     "median E diff (baseline - subject) [95% CI]": f"{t['energy_diff_baseline_minus_vqc']:.4g} [{t['energy_diff_ci'][0]:.4g}, {t['energy_diff_ci'][1]:.4g}]",
                     "energy win": t["energy_win"], "best baseline (success)": t["best_baseline_success"],
                     "success diff (subject - baseline) [95% CI]": f"{t['success_diff_vqc_minus_baseline']:.3g} [{t['success_diff_ci'][0]:.3g}, {t['success_diff_ci'][1]:.3g}]",
                     "success win": t["success_win"]})
    L += [md_table(pd.DataFrame(rows)), "",
          f"E7 (MPS chi=4) mean |<Z>_MPS - <Z>_exact| = {'n/a' if verdict['e7_mae_chi4'] is None else format(verdict['e7_mae_chi4'], '.4f')} (clause holds iff > {M.E7_TAU}).", "",
          "## Secondary: parity with the best classical PPO policy", "", md_table(pd.DataFrame(parity)), "",
          f"Parity (CI contains 0 or subject better) on every target: **{par_ok}**. This is not a claim of superiority.", "",
          "## Summary table (confirmation seeds)", ""]
    st = M.summary_table(df)
    for tid, g in st.groupby("target", sort=False):
        g = g.copy()
        g["median best energy [95% CI]"] = [f"{a_:.4g} [{b_:.4g}, {c_:.4g}]" for a_, b_, c_ in zip(g.median_best_energy, g.energy_ci_lo, g.energy_ci_hi)]
        g["success rate [95% CI]"] = [f"{a_:.2f} [{b_:.2f}, {c_:.2f}]" for a_, b_, c_ in zip(g.success_rate, g.success_ci_lo, g.success_ci_hi)]
        L += [f"### `{tid}`", "", md_table(g[["method", "n_seeds", "n_params", "median best energy [95% CI]", "success rate [95% CI]"]]), ""]
        curves = {m: M.load_curves("confirm", tid, m) for m in [*BASELINES, "ppo_vqc", subject] if (df[df.target == tid].method == m).any()}
        fig = ROOT / "results" / "figures" / f"V2_curves_{tid}.png"
        plots.plot_curves(curves, f"V2 confirmation {tid}", fig)
        L += [f"![V2 {tid}](results/figures/{fig.name})", ""]
    (ROOT / "REPORT_V2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / "verdict_v2.json").write_text(json.dumps({**verdict, "parity_all_targets": par_ok, "parity": parity}, indent=2, default=str))
    print(f"V2 VERDICT -> {verdict['branch']}; parity with best PPO-MLP on all targets: {par_ok}")


if __name__ == "__main__":
    main()
