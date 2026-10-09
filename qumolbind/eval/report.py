"""REPORT.md generation. Every number below is read from files under results/ (or computed here from those files);
nothing is hard-coded. Missing inputs are reported as 'not run', never filled in."""
from __future__ import annotations

import importlib.metadata as md
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qumolbind.eval import metrics as M
from qumolbind.eval import plots
from qumolbind.eval.stats import bootstrap_diff_ci, bootstrap_ci, mann_whitney

NL = chr(10)
ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = RES / "figures"
PKGS = ["torch", "numpy", "pennylane", "pennylane-lightning", "qiskit", "qiskit-aer", "openmm", "rdkit", "torch_geometric", "scikit-learn",
        "lightgbm", "cma", "gymnasium", "pandas", "scipy", "transformers"]


def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    if df is None or df.empty:
        return "_no data_"
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else (fmt.get(c, "{:.4g}").format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _git() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return h + (" (+uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"


def random_start_success(target: str, n: int = 500, seed: int = 0) -> tuple[float, int]:
    """(chance level, K): fraction of uniformly random torsion poses with RMSD < 2 A, and the number of torsions."""
    from qumolbind.sim.target import load_target

    t = load_target(target)
    rng = np.random.default_rng(seed)
    return float(np.mean([t.rmsd(t.ligand.randomize(rng)) < 2.0 for _ in range(n)])), t.ligand.K


def bench_markdown() -> str:
    b = _read(RES / "oracle_benchmark.csv")
    if b is None:
        return "_not run_ (`python scripts/benchmark_oracle.py`)"
    return md_table(b[["target", "n_atoms", "platform", "precision", "threads", "device", "evals_per_s", "n_evals"]], {"evals_per_s": "{:.1f}"})


def update_architecture(doc: Path = ROOT / "docs" / "ARCHITECTURE.md") -> None:
    """Fill the BENCH block of docs/ARCHITECTURE.md from results/oracle_benchmark.csv."""
    txt = doc.read_text(encoding="utf-8")
    a, b = txt.index("<!-- BENCH:START -->"), txt.index("<!-- BENCH:END -->")
    intro = ("Single process, full fast-oracle evaluation per pose (pair terms + GBn2 complex + GBn2 ligand + MMFF strain), "
             "measured on this machine:")
    body = NL.join([intro, "", bench_markdown(), ""])
    doc.write_text(txt[: a + len("<!-- BENCH:START -->")] + NL + body + txt[b:], encoding="utf-8")


def update_readme(exp: str, verdict: dict, df: pd.DataFrame, doc: Path = ROOT / "README.md") -> None:
    """Fill the RESULTS block of README.md with generated, provenance-linked statements only."""
    txt = doc.read_text(encoding="utf-8")
    a, b = txt.index("<!-- RESULTS:START -->"), txt.index("<!-- RESULTS:END -->")
    sizes = None if df.empty else df.groupby(["target", "method"]).size()
    n_seed = "n/a" if sizes is None else f"{int(sizes.min())}-{int(sizes.max())}"
    budget = "n/a" if df.empty else int(df.budget.iloc[0])
    targets = "n/a" if df.empty else ", ".join(sorted(df.target.unique()))
    failed = "" if not verdict["failed_conditions"] else " (" + "; ".join(verdict["failed_conditions"]) + ")"
    lines = [
        f"Latest generated report: experiment `{exp}` (B = {budget} oracle calls, seeds per cell: {n_seed}, targets: {targets}).",
        "",
        f"**Pre-registered criterion -> `{verdict['branch']}`**" + failed,
        "",
        f"All tables and figures: [REPORT.md](REPORT.md). Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')} from `results/`.",
    ]
    doc.write_text(txt[: a + len("<!-- RESULTS:START -->")] + NL + NL.join(lines) + NL + txt[b:], encoding="utf-8")


def build_report(exp: str) -> tuple[str, dict]:
    df = M.load_experiment(exp)
    cfg = json.loads((RES / f"config_{exp}.json").read_text()) if (RES / f"config_{exp}.json").exists() else {}
    e7 = _read(RES / f"E7_{exp}.csv")
    verdict = M.criterion(df, e7) if not df.empty else {"branch": "NO DEMONSTRATED ADVANTAGE", "failed_conditions": ["no E1 results"], "per_target": [], "n_target_wins": 0, "e7_mae_chi4": None, "e7_clause_holds": None}
    L: list[str] = []
    A = L.append

    # ---------------------------------------------------------------- header
    A(f"# QuMolBind report - experiment `{exp}`\n")
    A("_Generated by `scripts/make_report.py` from files in `results/` only. Re-run `make report` to regenerate._\n")
    A("## 0. Provenance\n")
    A(f"- generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}; git commit: `{_git()}`")
    vers = []
    for p in PKGS:
        try:
            vers.append(f"{p} {md.version(p)}")
        except md.PackageNotFoundError:
            pass
    A("- versions: " + ", ".join(vers))
    if not df.empty:
        tg = sorted(df.target.unique())
        A(f"- oracle-call budget B per run: {int(df.budget.iloc[0])}; seeds per cell: {df.groupby(['target', 'method']).size().min()}-{df.groupby(['target', 'method']).size().max()}; targets: {', '.join(tg)}")
        A(f"- hyper-parameter tuning: {'3 trials per method on separate tuning seeds (see results/tuning_%s.json)' % exp if df.get('tuned', pd.Series([False])).any() else 'NONE in this run (middle value of each 3-point grid; equal for all methods)'}")
        pc = df.groupby("method").n_params.median().astype(int)
        A("- trainable parameters of the mean network (median over runs): " + ", ".join(f"`{m}`={int(v)}" for m, v in pc.items()))
        if "hp_n_quantum_params" in df:
            q = df[df.method == "ppo_vqc"]
            if len(q):
                A(f"- VQC (variant i): quantum params {int(q.hp_n_quantum_params.iloc[0])}, classical mean-net params {int(q.hp_n_classical_mean_params.iloc[0])}; "
                  f"the critic (classical, 256-64-64-1) and the Gaussian log_std are classical in every method.")
    A("")

    # ---------------------------------------------------------------- verdict
    A("## 1. Pre-registered verdict (docs/PREREGISTRATION.md)\n")
    A(f"**Branch applied: `{verdict['branch']}`**\n")
    if verdict["failed_conditions"]:
        A("Conditions not met:\n" + "\n".join(f"- {c}" for c in verdict["failed_conditions"]) + "\n")
    rows = []
    for t in verdict["per_target"]:
        if "note" in t:
            rows.append({"target": t["target"], "note": t["note"]})
            continue
        rows.append({
            "target": t["target"], "n_seeds(vqc)": t["n_seeds_vqc"], "best baseline (energy)": t["best_baseline_energy"],
            "median E diff (baseline - vqc) [95% CI]": f"{t['energy_diff_baseline_minus_vqc']:.4g} [{t['energy_diff_ci'][0]:.4g}, {t['energy_diff_ci'][1]:.4g}]",
            "energy win": t["energy_win"], "best baseline (success)": t["best_baseline_success"],
            "success diff (vqc - baseline) [95% CI]": f"{t['success_diff_vqc_minus_baseline']:.3g} [{t['success_diff_ci'][0]:.3g}, {t['success_diff_ci'][1]:.3g}]",
            "success win": t["success_win"], "MWU p (energy vs best)": t["mwu_energy_vqc_vs_best"]["p"],
            "rank-biserial": t["mwu_energy_vqc_vs_best"]["rank_biserial"],
        })
    A(md_table(pd.DataFrame(rows)))
    A(f"\nTarget-level wins: {verdict['n_target_wins']}. E7 mean |<Z>_MPS(chi={M.E7_CHI}) - <Z>_exact| = "
      f"{'not run' if verdict['e7_mae_chi4'] is None else format(verdict['e7_mae_chi4'], '.4f')} (clause holds iff > {M.E7_TAU}).\n")
    if not df.empty and int(df.groupby(['target', 'method']).size().max()) < 5:
        A("> **Statistical power warning:** fewer than 5 seeds per cell. A percentile bootstrap over so few seeds cannot support any "
          "claim; this run demonstrates the pipeline, not a result.\n")

    # ---------------------------------------------------------------- E1
    A("## 2. E1 - main comparison (same env, state, seeds and oracle budget)\n")
    if df.empty:
        A("_not run_\n")
    else:
        st = M.summary_table(df)
        for tid, g in st.groupby("target", sort=False):
            chance, K = random_start_success(tid)
            A(f"### target `{tid}` (K = {K} torsions; chance-level success of a uniformly random torsion pose: {chance:.3f})\n")
            show = g.drop(columns=["target"]).copy()
            show["median best energy [95% CI]"] = [f"{a:.4g} [{b:.4g}, {c:.4g}]" for a, b, c in zip(g.median_best_energy, g.energy_ci_lo, g.energy_ci_hi)]
            show["success rate [95% CI]"] = [f"{a:.2f} [{b:.2f}, {c:.2f}]" for a, b, c in zip(g.success_rate, g.success_ci_lo, g.success_ci_hi)]
            A(md_table(show[["method", "n_seeds", "n_params", "median best energy [95% CI]", "success rate [95% CI]", "median_best_rmsd", "mean_wall_s"]], {"mean_wall_s": "{:.1f}", "median_best_rmsd": "{:.2f}"}))
            curves = {m: M.load_curves(exp, tid, m) for m in M.BASELINES + ["ppo_vqc"] if (df[(df.target == tid)].method == m).any()}
            f = FIG / f"E1_curves_{exp}_{tid}.png"
            plots.plot_curves(curves, f"E1 {tid}: median +- IQR over seeds", f)
            A(f"\n![E1 curves {tid}]({f.relative_to(ROOT).as_posix()})\n")
        # pairwise vs VQC
        pr = []
        for tid in sorted(df.target.unique()):
            v = df[(df.target == tid) & (df.method == "ppo_vqc")].best_score.to_numpy()
            for m in M.BASELINES:
                b = df[(df.target == tid) & (df.method == m)].best_score.to_numpy()
                if len(v) and len(b):
                    d = bootstrap_diff_ci(b, v)
                    mw = mann_whitney(v, b)
                    pr.append({"target": tid, "baseline": m, "median E (baseline - vqc) [95% CI]": f"{d[0]:.4g} [{d[1]:.4g}, {d[2]:.4g}]", "MWU p": mw["p"], "rank-biserial (vqc vs baseline)": mw["rank_biserial"]})
        A("**Pairwise: ppo_vqc vs each classical baseline (best energy; positive difference favours the VQC).**\n")
        A(md_table(pd.DataFrame(pr)) + "\n")

    # ---------------------------------------------------------------- ablations
    def ablation(title: str, methods: list[str], fname: str) -> None:
        A(f"## {title}\n")
        if df.empty or not df.method.isin(methods).any():
            A("_not run_\n")
            return
        st = M.summary_table(df[df.method.isin(methods)])
        st["median best energy [95% CI]"] = [f"{a:.4g} [{b:.4g}, {c:.4g}]" for a, b, c in zip(st.median_best_energy, st.energy_ci_lo, st.energy_ci_hi)]
        st["success rate"] = st.success_rate.map("{:.2f}".format)
        A(md_table(st[["target", "method", "n_seeds", "n_params", "median best energy [95% CI]", "success rate"]]) + "\n")
        for tid in sorted(df.target.unique()):
            curves = {m: M.load_curves(exp, tid, m) for m in methods if (df[df.target == tid].method == m).any()}
            plots.plot_curves(curves, f"{title.split(' - ')[0]} {tid}", FIG / f"{fname}_{exp}_{tid}.png")
            A(f"![{fname} {tid}]({(FIG / f'{fname}_{exp}_{tid}.png').relative_to(ROOT).as_posix()})\n")

    ablation("E2 - VQC without entanglement", ["ppo_vqc", "ppo_vqc_noent"], "E2")
    ablation("E3 - encoding / ansatz sensitivity (angle encoding needs a counted d->n projection)", ["ppo_vqc", "ppo_vqc_angle", "ppo_vqc_ryrz", "ppo_vqc_cz", "ppo_vqc_proj"], "E3")
    ablation("E5 - shot noise in the readout (straight-through: sampled forward pass, analytic gradient)", ["ppo_vqc", "ppo_vqc_shots4096", "ppo_vqc_shots1024", "ppo_vqc_shots256"], "E5")

    # ---------------------------------------------------------------- E4
    A("## E4 - barren-plateau diagnostics (gradient variance at random init)\n")
    e4 = _read(RES / f"E4_{exp}.csv")
    if e4 is None:
        A("_not run_\n")
    else:
        A(md_table(e4[["init", "n_qubits", "n_layers", "n_inits", "n_params", "var_mean_over_params", "var_median_over_params", "var_first_layer_q0"]]) + "\n")
        A("_Cost = mean over real environment states of <Z_0>; variance over >= 200 random inits. Entries of order 1e-34 (e.g. the first-layer qubit-0 parameter "
          "when L = 1) are exact structural zeros - that parameter does not influence the measured observable in that circuit - not a barren plateau._\n")
        plots.plot_barren(e4, FIG / f"E4_{exp}.png")
        A(f"![E4]({(FIG / f'E4_{exp}.png').relative_to(ROOT).as_posix()})\n")

    # ---------------------------------------------------------------- E6
    A("## E6 - depolarizing gate noise on the trained policy (evaluation-only)\n")
    e6 = _read(RES / f"E6_{exp}.csv")
    if e6 is None:
        A("_not run_\n")
    else:
        g = e6.groupby("p_depol").agg(median_best_score=("median_best_score", "median"), success_rate=("success_rate", "mean"),
                                       mean_symlog_energy_improvement=("mean_symlog_energy_improvement", "mean"),
                                       mean_abs_action_mean_shift=("mean_abs_action_mean_shift", "mean"), n_policies=("seed", "count")).reset_index()
        A(md_table(g) + "\n")
        plots.plot_lines(g.p_depol, {"action-mean shift": (g.mean_abs_action_mean_shift.to_numpy(), np.zeros(len(g)))}, "depolarizing p per 2-qubit gate",
                         "mean |noisy - noiseless| action mean", "E6", FIG / f"E6_{exp}.png", logx=True)
        A(f"![E6]({(FIG / f'E6_{exp}.png').relative_to(ROOT).as_posix()})\n")

    # ---------------------------------------------------------------- E7
    A("## E7 - classical simulability (Qiskit-Aer MPS)\n")
    if e7 is None:
        A("_not run_\n")
    else:
        g = e7.groupby("chi").agg(mean_abs_diff=("mae_mean_abs_diff", "mean"), max_abs_diff=("max_abs_diff", "max"), n_policies=("seed", "count")).reset_index()
        A(md_table(g) + "\n")
        ent = _read(RES / f"E7_entropy_{exp}.csv")
        if ent is not None:
            A("Output-state entanglement entropy (bits; middle cut of the 8-qubit state, max possible 4):\n")
            A(md_table(ent.groupby("which").agg(entropy_mid_cut_bits=("entropy_mid_cut_bits", "mean"), n=("seed", "count")).reset_index()) + "\n")

    # ---------------------------------------------------------------- E8
    A("## E8 - transfer (pretrain on A, fine-tune on B vs scratch on B, same fine-tune budget)\n")
    e8 = _read(RES / f"E8_{exp}.csv")
    if e8 is None:
        A("_not run_\n")
    else:
        rows = []
        for m, g in e8.groupby("method"):
            p = g[g["mode"] == "pretrained"].sort_values("seed").best_score.to_numpy()
            s = g[g["mode"] == "scratch"].sort_values("seed").best_score.to_numpy()
            d = bootstrap_diff_ci(s, p)
            rows.append({"method": m, "source->dest": f"{g.source.iloc[0]}->{g.dest.iloc[0]}", "finetune_budget": int(g.finetune_budget.iloc[0]), "n_seeds": len(p),
                         "median E scratch": float(np.median(s)), "median E pretrained": float(np.median(p)),
                         "median diff scratch - pretrained [95% CI]": f"{d[0]:.4g} [{d[1]:.4g}, {d[2]:.4g}]",
                         "success scratch": float(g[g['mode'] == 'scratch'].success.mean()), "success pretrained": float(g[g['mode'] == 'pretrained'].success.mean())})
        A(md_table(pd.DataFrame(rows)) + "\n")

    # ---------------------------------------------------------------- AL
    A("## Active learning / multi-fidelity (Level 0 surrogate, Level 1 fast oracle, Level 2 MD MM-GBSA-style)\n")
    al_dirs = sorted((RES / "al" / exp).glob("*_seed*"))
    if not al_dirs:
        A("_not run_\n")
    for d in al_dirs:
        rounds, cand = _read(d / "rounds.csv"), _read(d / "candidates.csv")
        A(f"### `{d.name}`\n")
        if rounds is not None:
            keep = [c for c in ["round", "oracle_calls_L1", "best_L1_score", "n_candidates", "selection", "n_L2_this_round", "n_L2_failed", "L2_md_ps", "L2_seconds", "n_L2_total", "L0_abs_err_target_units_on_new_L2"] if c in rounds]
            A(md_table(rounds[keep]) + "\n")
        if cand is not None:
            lab = cand[cand.selected_for_l2 & ~cand.l2_failed]
            if len(lab) >= 3:
                from scipy.stats import spearmanr

                A(f"L2-labelled candidates: {len(lab)}; Spearman(L1 score, L2 dG) = {spearmanr(lab.l1_score, lab.l2_dg).statistic:.2f}; "
                  f"L2 dG range [{lab.l2_dg.min():.1f}, {lab.l2_dg.max():.1f}] kJ/mol vs L1 score range [{lab.l1_score.min():.3g}, {lab.l1_score.max():.3g}] kJ/mol.\n")
        if (d / "calibration.png").exists():
            A(f"![calibration {d.name}]({(d / 'calibration.png').relative_to(ROOT).as_posix()})\n")

    # ---------------------------------------------------------------- hw / affinity
    A("## Ligand parametrisation (generated force field, D11)\n")
    pr = _read(RES / "parametrization_report.csv")
    A("_not run_\n" if pr is None else md_table(pr) + f"\n\nFailure rate: {100 * (1 - pr.ok.mean()):.0f}% ({int((~pr.ok).sum())}/{len(pr)}).\n")
    A("## Fast-oracle benchmark (measured by scripts/benchmark_oracle.py)\n")
    A(bench_markdown() + "\n")
    A("## Hardware feasibility: two-qubit gates after transpiling to a Heron-like device (cz/rz/sx/x, heavy-hex)\n")
    hw = _read(RES / "hw_cost.csv")
    A("_not run_\n" if hw is None else md_table(hw[["entangler", "n_qubits", "amplitude_dim", "cz_encoding_plus_ansatz", "depth_encoding_plus_ansatz", "cz_ansatz_only", "depth_ansatz_only"]]) + "\n")
    A("## Affinity models (scaffold split, test RMSE in pAffinity units)\n")
    aff = RES / "affinity_table.md"
    A(aff.read_text() if aff.exists() else "_not run_\n")
    return "\n".join(L) + "\n", verdict
