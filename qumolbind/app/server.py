"""Local web app: 3D pose viewer + every prototype function (oracle, search/PPO/QPPO runs, trained-policy stepping, results).

Run:  python scripts/serve.py   ->  http://127.0.0.1:8765
The server is local-only by default. Nothing here sends anything to quantum hardware.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem

from qumolbind.baselines import REGISTRY, TUNING_GRID
from qumolbind.baselines.common import Problem, make_problem
from qumolbind.env.state import build_state
from qumolbind.rl.budget import BudgetExhausted, Tracker
from qumolbind.utils.config import load_config, to_dict

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
RES = ROOT / "results"
METHODS = ["random_search", "hill_climb", "cmaes", "ppo_mlp_matched", "ppo_mlp_large", "ppo_vqc"]

app = FastAPI(title="QuMolBind")
_CFG = to_dict(load_config([]))
_problems: dict[str, Problem] = {}
_plock = threading.Lock()
_olock = threading.RLock()  # OpenMM contexts are not thread-safe: every oracle call goes through this lock
_jobs: dict[str, "Job"] = {}


def available_targets() -> list[str]:
    ids = {p.name for p in (ROOT / "data_cache" / "targets").glob("*") if (p / "protein.pdb").exists()}
    ids |= {p.name.split("_protein")[0] for p in (ROOT / "tests" / "fixtures").glob("*_protein.pdb")}
    return sorted(ids)


def get_problem(tid: str) -> Problem:
    with _plock:
        if tid not in available_targets():
            raise HTTPException(404, f"unknown target {tid}")
        if tid not in _problems:
            p = make_problem(tid, _CFG["env"])
            orig = p.oracle.evaluate
            p.oracle.evaluate = lambda c, _o=orig: _locked(_o, c)  # type: ignore[method-assign]
            _problems[tid] = p
        return _problems[tid]


def _locked(fn, c):
    with _olock:
        return fn(c)


def molblock(problem: Problem, coords: np.ndarray) -> str:
    m = Chem.Mol(problem.target.ligand.mol)
    conf = m.GetConformer()
    for i, p in enumerate(coords):
        conf.SetAtomPosition(i, [float(x) for x in p])
    return Chem.MolToMolBlock(m)


def terms_dict(t: Any) -> dict[str, float]:
    return {k: float(getattr(t, k)) for k in ("vdw", "elec", "solv", "strain", "e_int", "score")}


# --------------------------------------------------------------------------- targets / oracle
@app.get("/api/targets")
def targets() -> list[dict]:
    from qumolbind.sim.target import load_target

    out = []
    for t in available_targets():
        lm = load_target(t).ligand
        out.append({"id": t, "K": lm.K, "n_ligand_atoms": lm.n_atoms})
    return out


@app.get("/api/target/{tid}")
def target(tid: str) -> dict:
    p = get_problem(tid)
    lm = p.target.ligand
    return {
        "id": tid, "K": lm.K, "protein_pdb": Path(p.target.protein_pdb).read_text(), "native_mol": molblock(p, lm.native),
        "native_torsions": [float(x) for x in lm.get_torsions_deg(lm.native)],
        "torsion_atoms": [[int(t.i), int(t.a), int(t.b), int(t.l)] for t in lm.torsions],
        "platform": p.oracle.platform_name,
    }


class EvalReq(BaseModel):
    target: str
    torsions: list[float]


def _eval(p: Problem, torsions: list[float]) -> dict:
    lm = p.target.ligand
    coords = lm.set_torsions_deg(lm.native, np.asarray(torsions, dtype=float))
    t = p.oracle.evaluate(coords)
    return {"mol": molblock(p, coords), "terms": terms_dict(t), "rmsd": float(p.target.rmsd(coords)),
            "torsions": [float(x) for x in lm.get_torsions_deg(coords)]}


@app.post("/api/evaluate")
def evaluate(r: EvalReq) -> dict:
    p = get_problem(r.target)
    if len(r.torsions) != p.target.ligand.K:
        raise HTTPException(400, f"expected {p.target.ligand.K} torsions")
    return _eval(p, r.torsions)


@app.get("/api/randomize/{tid}")
def randomize(tid: str, seed: int | None = None) -> dict:
    p = get_problem(tid)
    lm = p.target.ligand
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % 2**32)
    return {"torsions": [float(x) for x in lm.get_torsions_deg(lm.randomize(rng))]}


# --------------------------------------------------------------------------- search / PPO / QPPO jobs
class LiveTracker(Tracker):
    """Tracker that records every improvement (for animation) and supports cancellation."""

    def __init__(self, *a, job: "Job", **k) -> None:
        super().__init__(*a, **k)
        self.job = job

    def evaluate(self, coords):
        if self.job.cancel:
            raise BudgetExhausted
        before = self.best_score
        et = super().evaluate(coords)
        if self.best_score < before:
            lm = self.job.problem.target.ligand
            self.job.history.append({"call": self.calls, "score": float(self.best_score), "rmsd": float(self.best_rmsd),
                                     "torsions": [float(x) for x in lm.get_torsions_deg(coords)], "terms": terms_dict(et)})
        return et


class LiveProblem(Problem):
    job: "Job" = None  # type: ignore[assignment]

    def tracker(self, budget: int) -> Tracker:  # type: ignore[override]
        self.job.tracker = LiveTracker(self.oracle, budget, self.target.rmsd, job=self.job)
        return self.job.tracker


class Job:
    def __init__(self, req: "JobReq", problem: Problem) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.req, self.problem = req, problem
        self.status, self.error, self.cancel = "running", "", False
        self.tracker: LiveTracker | None = None
        self.history: list[dict] = []
        self.result: dict | None = None
        self.t0 = time.time()


class JobReq(BaseModel):
    target: str
    method: str
    budget: int = 1000
    seed: int = 0
    hp: float | None = None  # lr for PPO methods, step (deg) for hill_climb, sigma0 (deg) for cmaes


def _run_job(job: Job) -> None:
    r = job.req
    try:
        base = get_problem(r.target)
        lp = LiveProblem(base.target_id, base.target, base.oracle, base.env_cfg)
        lp.job = job
        job.problem = lp
        if r.method == "ppo_vqc":
            from qumolbind.quantum.runner import make_variant_runner

            fn = make_variant_runner("ppo_vqc")
        else:
            fn = REGISTRY[r.method]
        axis = TUNING_GRID[r.method][0]
        kw: dict[str, Any] = {}
        if r.hp is not None and axis != "tune_seed":
            kw[axis] = r.hp
        if r.method.startswith("ppo"):
            kw["actor_cfg"] = {k: v for k, v in _CFG["actor"].items() if k != "lr"}
        res = fn(lp, r.budget, r.seed, **kw)
        job.result = {"best_score": res.best_score, "best_rmsd": res.best_rmsd, "success": bool(res.success), "calls_used": res.calls_used,
                      "n_params": res.n_params, "wall_s": res.wall_s}
        job.status = "cancelled" if job.cancel else "done"
    except Exception as e:  # noqa: BLE001
        job.status, job.error = "error", f"{type(e).__name__}: {e}"


@app.post("/api/job/start")
def job_start(r: JobReq) -> dict:
    if r.method not in METHODS:
        raise HTTPException(400, f"method must be one of {METHODS}")
    if any(j.status == "running" for j in _jobs.values()):
        raise HTTPException(409, "another job is running; cancel it first")
    r.budget = int(min(max(r.budget, 50), 20000))
    job = Job(r, get_problem(r.target))
    _jobs[job.id] = job
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"id": job.id}


@app.get("/api/job/{jid}")
def job_state(jid: str, since: int = 0) -> dict:
    j = _jobs.get(jid)
    if j is None:
        raise HTTPException(404)
    tr = j.tracker
    curve = [] if tr is None else [float(x) for x in tr.curve[: tr.calls : max(tr.calls // 400, 1)]]
    new = j.history[since:]
    lm = j.problem.target.ligand
    for h in new:  # attach a mol block per new improvement so the viewer can animate without extra requests
        if "mol" not in h:
            h["mol"] = molblock(j.problem, lm.set_torsions_deg(lm.native, np.asarray(h["torsions"])))
    return {"status": j.status, "error": j.error, "calls": 0 if tr is None else tr.calls, "budget": j.req.budget, "curve": curve,
            "best_score": None if tr is None or not np.isfinite(tr.best_score) else float(tr.best_score),
            "best_rmsd": None if tr is None or not np.isfinite(tr.best_rmsd) else float(tr.best_rmsd),
            "new_history": new, "n_history": len(j.history), "result": j.result, "elapsed_s": time.time() - j.t0}


@app.post("/api/job/{jid}/cancel")
def job_cancel(jid: str) -> dict:
    j = _jobs.get(jid)
    if j is None:
        raise HTTPException(404)
    j.cancel = True
    return {"ok": True}


# --------------------------------------------------------------------------- trained policies
@app.get("/api/policies")
def policies() -> list[dict]:
    out = []
    for p in sorted(RES.glob("*/ckpt/*_ppo_vqc*.pt")):
        tid = p.name.split("_")[0]
        out.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"), "target": tid, "experiment": p.parent.parent.name, "name": p.name})
    return out


class ActReq(BaseModel):
    target: str
    ckpt: str
    torsions: list[float]
    t: int = 0
    deterministic: bool = True


@app.post("/api/policy/act")
def policy_act(r: ActReq) -> dict:
    """One policy step from the given pose: state -> VQC mean -> action -> new torsions (and the evaluated new pose)."""
    from qumolbind.eval.policy_io import load_vqc_actor

    p = get_problem(r.target)
    ck = (ROOT / r.ckpt).resolve()
    if RES.resolve() not in ck.parents or not ck.exists():
        raise HTTPException(400, "checkpoint must be under results/")
    method = "_".join(ck.stem.split("_")[1:-1]) or "ppo_vqc"
    actor = load_vqc_actor(ck, p, _CFG["actor"], method=method if method.startswith("ppo_vqc") else "ppo_vqc")
    lm = p.target.ligand
    coords = lm.set_torsions_deg(lm.native, np.asarray(r.torsions))
    terms = p.oracle.evaluate(coords)
    st = build_state(lm.get_torsions_deg(coords), terms.terms(), r.t, _CFG["env"]["episode_len"], _CFG["env"]["state_dim"])
    with torch.no_grad():
        obs = torch.as_tensor(st)[None]
        mean = actor(obs)[0]
        act = mean if r.deterministic else mean + torch.randn_like(mean) * actor.log_std.exp()
    act = act.clamp(-1, 1).numpy()
    new = lm.apply_torsion_deltas(coords, act * _CFG["env"]["max_delta_deg"])
    out = _eval(p, [float(x) for x in lm.get_torsions_deg(new)])
    out.update({"action_mean": [float(x) for x in mean.numpy()], "action": [float(x) for x in act], "before_score": float(terms.score),
                "n_quantum_params": actor.n_quantum_params()})
    return out


# --------------------------------------------------------------------------- results
@app.get("/api/results")
def results() -> dict:
    from qumolbind.eval.metrics import load_experiment, summary_table

    exps, tables = [], {}
    for d in sorted(RES.glob("*/rows")):
        name = d.parent.name
        df = load_experiment(name)
        if df.empty:
            continue
        exps.append(name)
        st = summary_table(df)
        tables[name] = st.replace({np.nan: None}).to_dict(orient="records")
    figs = sorted(str(p.relative_to(RES)).replace("\\", "/") for p in (RES / "figures").glob("*.png")) if (RES / "figures").exists() else []
    aux = {}
    for n in ("hw_cost", "oracle_benchmark", "parametrization_report"):
        f = RES / f"{n}.csv"
        if f.exists():
            aux[n] = pd.read_csv(f).replace({np.nan: None}).to_dict(orient="records")
    return {"experiments": exps, "tables": tables, "figures": figs, "aux": aux}


@app.get("/api/report", response_class=PlainTextResponse)
def report() -> str:
    f = ROOT / "REPORT.md"
    return f.read_text(encoding="utf-8") if f.exists() else "REPORT.md not generated yet (run `make report`)."


@app.get("/api/al")
def al_runs() -> list[dict]:
    out = []
    for d in sorted((RES / "al").glob("*/*_seed*")):
        out.append({"id": f"{d.parent.name}/{d.name}", "has_calibration": (d / "calibration.png").exists()})
    return out


@app.get("/api/al/{exp}/{run}")
def al_run(exp: str, run: str) -> dict:
    d = RES / "al" / exp / run
    if not d.exists():
        raise HTTPException(404)
    rd = lambda n: pd.read_csv(d / n).replace({np.nan: None}).to_dict(orient="records") if (d / n).exists() else []  # noqa: E731
    return {"rounds": rd("rounds.csv"), "candidates": rd("candidates.csv"), "calibration": f"al/{exp}/{run}/calibration.png"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/results", StaticFiles(directory=str(RES)), name="results")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
