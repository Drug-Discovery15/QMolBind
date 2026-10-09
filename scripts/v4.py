"""V4 (docs/PREREGISTRATION_V4.md): greedy-acceptance + elite-restart learned local search. Phases: explore | confirm.

  python scripts/v4.py explore     # seeds 500,501 on 3ert+1uyd, 6 trials per PPO method -> results/explore4_selection.json
  python scripts/v4.py confirm     # seeds 600..609 on 3 targets -> REPORT_V4.md (advantage verdict + parity verdict)
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.baselines import REGISTRY  # noqa: E402
from qumolbind.baselines.common import make_problem  # noqa: E402
from qumolbind.env.state import symlog  # noqa: E402
from qumolbind.eval import metrics as M  # noqa: E402
from qumolbind.eval import plots  # noqa: E402
from qumolbind.eval.policy_io import load_vqc_actor, rollout  # noqa: E402
from qumolbind.eval.report import md_table  # noqa: E402
from qumolbind.eval.stats import bootstrap_diff_ci  # noqa: E402
from qumolbind.quantum.encoding import amplitude_state  # noqa: E402
from qumolbind.quantum.mps import mps_expectations, mps_expectations_reupload  # noqa: E402
from qumolbind.utils.config import load_config, to_dict  # noqa: E402
from scripts.explore_vqc import SETTINGS, registry  # noqa: E402

LRS = [3e-3, 1e-2, 3e-2]
INIT_LS = [-1.6, -0.9]
ELITE_P = 0.9
METHODS = ["ppo_zero_mean", "ppo_mlp_matched", "ppo_mlp_large", "ppo_mlp_matched_reup", "ppo_vqc", "ppo_vqc_aff", "ppo_vqc_reup"]
VQC = ["ppo_vqc", "ppo_vqc_aff", "ppo_vqc_reup"]
SEARCH = ["random_search", "hill_climb", "cmaes"]
BASELINES = [*SEARCH, "ppo_zero_mean", "ppo_mlp_matched", "ppo_mlp_large", "ppo_mlp_matched_reup"]
EXP_SEEDS, EXP_TARGETS = [500, 501], ["3ert", "1uyd"]
import os  # noqa: E402
_a, _b = (os.environ.get("QMB_V4_SEEDS", "600:610")).split(":")
CONF_SEEDS, CONF_TARGETS = list(range(int(_a), int(_b))), ["3ert", "1uyd", "1eve"]
CONF_DIR = os.environ.get("QMB_V4_DIR", "confirm4")  # V5 = replication of the V4 protocol on fresh seeds: QMB_V4_DIR=confirm5 QMB_V4_SEEDS=700:730
REPORT_NAME = os.environ.get("QMB_V4_REPORT", "REPORT_V4.md")
B = 2000
MARGIN = 100.0   # kJ/mol non-inferiority margin (pre-registered)
SETTING = SETTINGS["A"]


def v4_problem(base):
    return replace(base, env_cfg={**base.env_cfg, "elite_start_prob": ELITE_P, "greedy_accept": True})


def call(fn, name, prob, seed, lr, ils, actor_cfg, ckpt=None):
    kw = dict(lr=lr, ppo_kw=dict(SETTING))
    if name == "ppo_zero_mean" or name == "ppo_mlp_large":
        kw["init_log_std"] = ils
    elif name.startswith("ppo_vqc"):
        kw["actor_cfg"] = {**actor_cfg, "init_log_std": ils}
    else:
        kw.update(actor_cfg=actor_cfg, init_log_std=ils)
    if ckpt:
        kw["ckpt_path"] = str(ckpt)
    return fn(prob, B, seed, **kw)


def explore() -> None:
    d = to_dict(load_config(["experiment=pilot"]))
    actor_cfg = {k: v for k, v in d["actor"].items() if k != "lr"}
    out = ROOT / "results" / "explore4" / "rows"
    out.mkdir(parents=True, exist_ok=True)
    reg = {m: f for m, f in registry().items() if m in METHODS}
    for tid in EXP_TARGETS:
        prob = v4_problem(make_problem(tid, d["env"]))
        print(f"== explore4 target {tid}", flush=True)
        for name, fn in reg.items():
            for lr, ils in product(LRS, INIT_LS):
                for s in EXP_SEEDS:
                    f = out / f"{tid}_{name}_lr{lr:g}_ls{ils:g}_s{s}.json"
                    if f.exists():
                        continue
                    r = call(fn, name, prob, s, lr, ils, actor_cfg)
                    f.write_text(json.dumps({"target": tid, "method": name, "lr": lr, "init_log_std": ils, "seed": s, "best_score": r.best_score,
                                             "best_rmsd": r.best_rmsd, "success": bool(r.success), "n_params": r.n_params}))
                    print(f"  {name:22s} lr={lr:g} ls={ils:g} seed={s} best={r.best_score:10.1f} ({r.wall_s:.0f}s)", flush=True)
    rows = [json.loads(p.read_text()) for p in out.glob("*.json")]
    sel: dict = {"configs": {}, "scores": {}}
    for name in METHODS:
        best, best_sc, table = None, np.inf, {}
        for lr, ils in product(LRS, INIT_LS):
            cell = [r for r in rows if r["method"] == name and r["lr"] == lr and r["init_log_std"] == ils]
            if len(cell) < len(EXP_SEEDS) * len(EXP_TARGETS):
                continue
            sc = float(np.mean([symlog(r["best_score"], 100.0) for r in cell]))
            table[f"lr={lr:g},ls={ils:g}"] = sc
            if sc < best_sc:
                best, best_sc = {"lr": lr, "init_log_std": ils}, sc
        sel["configs"][name], sel["scores"][name] = best, {"best": best_sc, "all": table}
    sel["subject"] = min(VQC, key=lambda v: sel["scores"][v]["best"])
    (ROOT / "results" / "explore4_selection.json").write_text(json.dumps(sel, indent=2))
    print("\nV4 exploration scores (mean symlog(best/100), lower is better):")
    for n, sc in sorted(sel["scores"].items(), key=lambda kv: kv[1]["best"]):
        print(f"  {n:24s} {sc['best']:.3f}  config={sel['configs'][n]}")
    print("SUBJECT:", sel["subject"])


def confirm() -> None:
    sel = json.loads((ROOT / "results" / "explore4_selection.json").read_text())
    subject = sel["subject"]
    tuning = json.loads((ROOT / "results" / "tuning_pilot.json").read_text())
    d = to_dict(load_config(["experiment=pilot"]))
    actor_cfg = {k: v for k, v in d["actor"].items() if k != "lr"}
    base = ROOT / "results" / CONF_DIR
    for sub in ("rows", "ckpt", "curves"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    preg = registry()
    methods = [*BASELINES, subject]
    for tid in CONF_TARGETS:
        bprob = make_problem(tid, d["env"])
        prob = v4_problem(bprob)
        print(f"== confirm4 target {tid}", flush=True)
        for m in methods:
            for s in CONF_SEEDS:
                f = base / "rows" / f"{tid}_{m}_{s}.json"
                if f.exists():
                    continue
                if m in SEARCH:
                    r = REGISTRY[m](bprob, B, s, **tuning.get(f"{tid}/{m}", {}).get("hparams", {}))
                else:
                    c = sel["configs"][m]
                    r = call(preg[m], m, prob, s, c["lr"], c["init_log_std"], actor_cfg, ckpt=base / "ckpt" / f"{tid}_{m}_{s}.pt" if m == subject else None)
                np.save(base / "curves" / f"{tid}_{m}_{s}.npy", np.stack([r.curve, r.rmsd_curve]))
                row = r.row(); row["method"] = m
                f.write_text(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}))
                print(f"  {m:22s} seed={s} best={r.best_score:10.1f} rmsd={r.best_rmsd:4.2f} params={r.n_params} ({r.wall_s:.0f}s)", flush=True)
    df = pd.DataFrame([json.loads(p.read_text()) for p in sorted((base / "rows").glob("*.json"))])

    # E7 clause on the subject
    e7_rows = []
    for tid in CONF_TARGETS:
        bprob = make_problem(tid, d["env"])
        for s in CONF_SEEDS:
            ck = base / "ckpt" / f"{tid}_{subject}_{s}.pt"
            if not ck.exists():
                continue
            actor = load_vqc_actor(ck, bprob, d["actor"], method=subject)
            states, *_ = rollout(bprob, lambda o: actor(o), actor.log_std.detach(), 6, seed=60_000 + s)
            x = torch.as_tensor(states[np.random.default_rng(s).permutation(len(states))[:24]], dtype=torch.float64)
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
    e7.to_csv(ROOT / "results" / f"{CONF_DIR}_E7.csv", index=False)
    verdict = M.criterion(df, e7 if len(e7) else None, baselines=BASELINES, vqc=subject)

    # parity (non-inferiority) verdict + diagnostic vs ppo_zero_mean
    par_rows, par_ok = [], True
    for tid in CONF_TARGETS:
        g = df[df.target == tid]
        sub = g[g.method == subject]
        base_m = {m: g[g.method == m] for m in BASELINES if (g.method == m).any()}
        bm = min(base_m, key=lambda m: base_m[m].best_score.median())
        de = bootstrap_diff_ci(base_m[bm].best_score.to_numpy(), sub.best_score.to_numpy())             # >0: subject better
        ds = bootstrap_diff_ci(base_m[bm].success.astype(float).to_numpy(), sub.success.astype(float).to_numpy(), np.mean)  # >0: subject worse
        ok = bool(de[1] >= -MARGIN and ds[2] < 0.3)
        par_ok &= ok
        z = g[g.method == "ppo_zero_mean"]
        dz = bootstrap_diff_ci(z.best_score.to_numpy(), sub.best_score.to_numpy())
        par_rows.append({"target": tid, "best baseline": bm, "median E diff (baseline - subject) [95% CI]": f"{de[0]:.4g} [{de[1]:.4g}, {de[2]:.4g}]",
                         f"CI lower >= -{MARGIN:g}": bool(de[1] >= -MARGIN), "success diff (baseline - subject) upper CI": round(ds[2], 3), "parity here": ok,
                         "vs ppo_zero_mean: median E diff [95% CI]": f"{dz[0]:.4g} [{dz[1]:.4g}, {dz[2]:.4g}]"})
    parity = "PARITY" if par_ok else "NOT PARITY"

    L = ["# QuMolBind V4 report - greedy-acceptance learned local search", "",
         "_Generated by `scripts/v4.py confirm`; protocol fixed in `docs/PREREGISTRATION_V4.md` before any V4 run._", "",
         f"- subject (pre-registered selection rule): **`{subject}`**; seeds {CONF_SEEDS[0]}-{CONF_SEEDS[-1]}, targets {', '.join(CONF_TARGETS)}, B = {B}",
         "- configs: " + ", ".join(f"`{m}`=lr {c['lr']:g}, init log-std {c['init_log_std']:g}" for m, c in sel["configs"].items()), "",
         "## Advantage verdict (mechanical)", "", f"**`{verdict['branch']}`**", ""]
    L += ["Conditions not met:", *[f"- {c}" for c in verdict["failed_conditions"]], ""] if verdict["failed_conditions"] else []
    ar = [{"target": t["target"], "best baseline (energy)": t["best_baseline_energy"],
           "median E diff (baseline - subject) [95% CI]": f"{t['energy_diff_baseline_minus_vqc']:.4g} [{t['energy_diff_ci'][0]:.4g}, {t['energy_diff_ci'][1]:.4g}]",
           "energy win": t["energy_win"], "success win": t["success_win"]} for t in verdict["per_target"] if "note" not in t]
    L += [md_table(pd.DataFrame(ar)), "",
          f"E7 (MPS chi=4) mean |<Z>_MPS - <Z>_exact| = {'n/a' if verdict['e7_mae_chi4'] is None else format(verdict['e7_mae_chi4'], '.4f')} (clause holds iff > {M.E7_TAU}).", "",
          f"## Parity verdict (non-inferiority margin {MARGIN:g} kJ/mol): **`{parity}`**", "", md_table(pd.DataFrame(par_rows)), "",
          "The last column is the diagnostic: if the subject is not better than `ppo_zero_mean`, its parity says nothing about the mean network.", "",
          "## Summary table", ""]
    st = M.summary_table(df)
    for tid, g in st.groupby("target", sort=False):
        g = g.copy()
        g["median best energy [95% CI]"] = [f"{a:.4g} [{b:.4g}, {c:.4g}]" for a, b, c in zip(g.median_best_energy, g.energy_ci_lo, g.energy_ci_hi)]
        g["success rate [95% CI]"] = [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(g.success_rate, g.success_ci_lo, g.success_ci_hi)]
        L += [f"### `{tid}`", "", md_table(g[["method", "n_seeds", "n_params", "median best energy [95% CI]", "success rate [95% CI]"]]), ""]
        curves = {m: M.load_curves(CONF_DIR, tid, m) for m in methods if (df[df.target == tid].method == m).any()}
        fig = ROOT / "results" / "figures" / f"V4_curves_{tid}.png"
        plots.plot_curves(curves, f"V4 confirmation {tid}", fig)
        L += [f"![V4 {tid}](results/figures/{fig.name})", ""]
    (ROOT / REPORT_NAME).write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / f"verdict_{CONF_DIR}.json").write_text(json.dumps({**verdict, "parity": parity, "parity_rows": par_rows}, indent=2, default=str))
    print(f"V4 VERDICT -> advantage: {verdict['branch']}; parity: {parity}")


if __name__ == "__main__":
    {"explore": explore, "confirm": confirm}[sys.argv[1]]()
