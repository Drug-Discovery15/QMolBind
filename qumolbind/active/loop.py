"""Active-learning loop with multi-fidelity scoring.

Per round:  (1) the policy keeps training with PPO against the Level-1 oracle (budget share B/rounds);
            (2) the poses it visited this round with L1 score < ``max_l1_score`` are reduced to M diverse candidates;
            (3) the Level-0 surrogate ensemble scores them (mu, sigma) BEFORE any Level-2 label exists for them;
            (4) UCB picks the top-b for the Level-2 oracle (short MD + MM-GBSA-style dG);
            (5) results are appended to the dataset, the surrogate is refit on all L2 labels and the critic is refit
                (value regression on the accumulated rollout returns); (6) the policy continues training.
Round 0 has no labels, so its candidates are chosen by lowest L1 score (warm start) and carry no L0 prediction.
Every logged number carries its fidelity (L0 surrogate / L1 fast oracle / L2 MD).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qumolbind.active.multifidelity import MultiFidelityDataset, Record
from qumolbind.active.surrogate import SurrogateEnsemble, pose_features, to_target
from qumolbind.active.ucb import select_top_b, ucb_scores
from qumolbind.baselines.common import Problem, finalize
from qumolbind.baselines.ppo_mlp import build_vec_env
from qumolbind.quantum.runner import make_vqc_actor, quantum_diagnostics
from qumolbind.rl.actors import MLPActor
from qumolbind.rl.critic import Critic
from qumolbind.rl.ppo import PPO, PPOConfig
from qumolbind.sim.oracle_slow import SlowOracle
from qumolbind.utils.logging import RunLogger
from qumolbind.utils.seeding import seed_everything


def circular_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(np.sqrt(np.mean(d**2)))


class ActiveLearningLoop:
    def __init__(self, problem: Problem, budget: int, al_cfg: dict, actor_cfg: dict | None, lr: float, seed: int,
                 out_dir: str | Path, ppo_kw: dict | None = None, actor_kind: str = "vqc") -> None:
        seed_everything(seed)
        self.problem, self.budget, self.cfg, self.seed = problem, budget, dict(al_cfg), seed
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.lm = problem.target.ligand
        self.tracker = problem.tracker(budget)
        self.tracker.log_poses = True
        cfg = PPOConfig(actor_lr=lr, seed=seed, **(ppo_kw or {}))
        self.vec = build_vec_env(problem, self.tracker, cfg.n_envs)
        d, K = problem.env_cfg.get("state_dim", 256), self.lm.K
        acfg = {k: v for k, v in (actor_cfg or {}).items() if k != "lr"}
        self.actor = make_vqc_actor(d, K, acfg) if actor_kind == "vqc" else MLPActor(d, K, (128, 128))
        self.critic = Critic(d)
        self.ppo = PPO(self.vec, self.actor, self.critic, cfg, logger=RunLogger(self.out, "ppo_metrics", tensorboard=False), on_update=self._on_update)
        self.slow = SlowOracle(problem.target.protein_pdb, self.lm.mol, self.lm.native, fast_oracle=problem.oracle, ligand_model=self.lm, platform=problem.env_cfg.get("oracle_platform", "auto"))
        self.surrogate = SurrogateEnsemble(self.cfg.get("ensemble", 5), seed=seed)
        self.data = MultiFidelityDataset()
        self.replay: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.rounds_log: list[dict] = []
        self._log_start = 0
        self._cand_id = 0

    # ------------------------------------------------------------------ hooks
    def _on_update(self, ppo: PPO, row: dict) -> None:
        self.replay.append((ppo.last_batch_obs.clone(), ppo.last_batch_ret.clone()))
        self.replay = self.replay[-60:]
        if hasattr(ppo.actor, "vqc"):
            quantum_diagnostics(ppo, row)

    def _candidates(self, r: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """M diverse low-L1-energy poses visited by the policy during this round."""
        log = self.tracker.pose_log[self._log_start :]
        self._log_start = len(self.tracker.pose_log)
        ok = [(c, t) for c, t in log if t[4] < self.cfg.get("max_l1_score", 1e4)]
        ok.sort(key=lambda ct: ct[1][4])
        chosen: list[tuple[np.ndarray, np.ndarray]] = []
        tors: list[np.ndarray] = []
        for c, t in ok:
            tt = self.lm.get_torsions_deg(c)
            if all(circular_dist(tt, o) > self.cfg.get("min_sep_deg", 15.0) for o in tors):
                chosen.append((c, t)); tors.append(tt)
            if len(chosen) >= self.cfg.get("M", 32):
                break
        return chosen

    def _retrain_critic(self, epochs: int = 5) -> float:
        if not self.replay:
            return float("nan")
        X = torch.cat([o for o, _ in self.replay])
        Y = torch.cat([y for _, y in self.replay])
        opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        loss = torch.tensor(float("nan"))
        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 128):
                idx = perm[s : s + 128]
                opt.zero_grad()
                loss = 0.5 * ((self.critic(X[idx]) - Y[idx]) ** 2).mean()
                loss.backward()
                opt.step()
        return float(loss.detach())

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        R, b, beta, md_ps = self.cfg.get("rounds", 2), self.cfg.get("b", 4), self.cfg.get("beta", 1.0), self.cfg.get("md_ps", 5.0)
        b = min(b, 50)  # hard cap: <= 50 Level-2 candidates per round
        t_start = time.time()
        for r in range(R):
            round_end = int(self.budget * (r + 1) / R)
            self.ppo.train(seed=self.seed + 1000 * r, until=lambda: self.tracker.calls >= round_end)
            cands = self._candidates(r)
            row = {"round": r, "oracle_calls_L1": self.tracker.calls, "best_L1_score": float(self.tracker.best_score),
                   "n_candidates": len(cands), "n_L2_total_before": len(self.data.labelled())}
            if not cands:
                row["note"] = "no candidate below max_l1_score"
                self.rounds_log.append(row)
                continue
            recs = []
            for c, t in cands:
                f = pose_features(self.lm.get_torsions_deg(c), t[:4])
                recs.append(self.data.add(Record(r, self._cand_id, f, l1_score=float(t[4]), l1_e_int=float(t[0] + t[1] + t[2]))))
                self._cand_id += 1
            X = np.stack([x.features for x in recs])
            mu_t, sd_t = self.surrogate.predict(X)
            mu_dg, sd_dg = self.surrogate.predict_dg(X)
            warm = not self.surrogate.members
            for k, x in enumerate(recs):
                if not warm:
                    x.l0_mu_t, x.l0_sigma_t, x.l0_mu_dg, x.l0_sigma_dg = float(mu_t[k]), float(sd_t[k]), float(mu_dg[k]), float(sd_dg[k])
                    x.acquisition = float(ucb_scores(mu_t[k : k + 1], sd_t[k : k + 1], beta)[0])
            sel = np.argsort([x.l1_score for x in recs])[:b] if warm else select_top_b(mu_t, sd_t, b, beta)
            row["selection"] = "warm_start_lowest_L1" if warm else f"UCB(beta={beta})"
            fails, secs = 0, 0.0
            for k in sel:
                x = recs[k]
                x.selected_for_l2 = True
                res = self.slow.delta_g(cands[k][0], md_ps=md_ps, seed=self.seed * 997 + x.cand_id)
                x.l2_dg, x.l2_sem, x.l2_seconds = res.dg, res.dg_sem, res.seconds
                x.l2_failed = not np.isfinite(res.dg)
                fails += int(x.l2_failed); secs += res.seconds
            X_l, y_l = self.data.xy()
            if len(y_l) >= 2:
                self.surrogate.fit(X_l, y_l)
            crit_loss = self._retrain_critic()
            lab = [x for x in recs if x.selected_for_l2 and not x.l2_failed]
            pre = [x for x in lab if np.isfinite(x.l0_mu_dg)]
            row.update({
                "n_L2_this_round": len(sel), "n_L2_failed": fails, "L2_md_ps": md_ps, "L2_seconds": secs,
                "n_L2_total": len(self.data.labelled()), "critic_loss_after_refit": crit_loss,
                "surrogate_train_n": self.surrogate.n_train,
                "L0_abs_err_target_units_on_new_L2": float(np.mean([abs(x.l0_mu_t - to_target(np.array([x.l2_dg]))[0]) for x in pre])) if pre else float("nan"),
                "mean_L1_of_selected": float(np.mean([x.l1_score for x in lab])) if lab else float("nan"),
                "mean_L2_of_selected": float(np.mean([x.l2_dg for x in lab])) if lab else float("nan"),
            })
            self.rounds_log.append(row)
            print(f"[AL] round {r}: {row}", flush=True)
        return self._finish(time.time() - t_start)

    # ------------------------------------------------------------------ outputs
    def _cv_predictions(self, folds: int = 5) -> pd.DataFrame:
        """Out-of-fold surrogate predictions on all L2-labelled records (for the calibration plot)."""
        X, y = self.data.xy()
        n = len(y)
        if n < 4:
            return pd.DataFrame(columns=["mu", "sigma", "actual"])
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(n)
        rows = []
        for f in range(min(folds, n)):
            te = order[f::folds]
            tr = np.setdiff1d(order, te)
            if len(tr) < 2:
                continue
            m = SurrogateEnsemble(self.surrogate.n_members, seed=self.seed + f).fit(X[tr], y[tr])
            mu, sd = m.predict_dg(X[te])
            rows += [{"mu": a, "sigma": s, "actual": t} for a, s, t in zip(mu, sd, y[te])]
        return pd.DataFrame(rows)

    def _finish(self, seconds: float) -> dict:
        df = self.data.to_frame()
        df.to_csv(self.out / "candidates.csv", index=False)
        pd.DataFrame(self.rounds_log).to_csv(self.out / "rounds.csv", index=False)
        cv = self._cv_predictions()
        cv.to_csv(self.out / "calibration_oof.csv", index=False)
        self._plot_calibration(df, cv)
        res = finalize(self.tracker, "ppo_vqc_al", self.problem.target_id, self.seed, self.actor.n_mean_params(), {}, seconds)
        sp = float("nan")
        lab = df[df.selected_for_l2 & ~df.l2_failed]
        if len(lab) >= 3:
            from scipy.stats import spearmanr

            sp = float(spearmanr(lab.l1_score, lab.l2_dg).statistic)
        summary = {"result": res, "n_L2": int(len(lab)), "spearman_L1_vs_L2": sp, "rounds": self.rounds_log}
        return summary

    def _plot_calibration(self, df: pd.DataFrame, cv: pd.DataFrame) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        pre = df[df.selected_for_l2 & ~df.l2_failed & df.l0_mu_dg.notna()]
        panels = [("Prospective: prediction made before the L2 label existed", pre.l0_mu_dg, pre.l0_sigma_dg, pre.l2_dg),
                  ("Out-of-fold (5-fold CV on all L2 labels)", cv.mu, cv.sigma, cv.actual)]
        for a, (title, mu, sd, act) in zip(ax, panels):
            if len(act):
                a.errorbar(act, mu, yerr=sd, fmt="o", ms=4, capsize=2, color="#2a6fbb")
                lo, hi = float(min(act.min(), mu.min())), float(max(act.max(), mu.max()))
                a.plot([lo, hi], [lo, hi], "k--", lw=1)
            a.set_xlabel("actual L2 dG (kJ/mol)  [fidelity L2: MD MM-GBSA-style]")
            a.set_ylabel("surrogate prediction (kJ/mol)  [fidelity L0]")
            a.set_title(f"{title}\nn={len(act)}", fontsize=9)
        fig.tight_layout()
        fig.savefig(self.out / "calibration.png", dpi=130)
        plt.close(fig)
