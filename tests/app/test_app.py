import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from qumolbind.app.server import app  # noqa: E402

client = TestClient(app)
TID = "3ert"


def test_targets_and_target_payload() -> None:
    ts = client.get("/api/targets").json()
    assert any(t["id"] == TID for t in ts)
    d = client.get(f"/api/target/{TID}").json()
    assert d["K"] == len(d["native_torsions"]) == len(d["torsion_atoms"]) and "ATOM" in d["protein_pdb"] and "V2000" in d["native_mol"]


def test_evaluate_matches_oracle_and_native_rmsd_zero() -> None:
    d = client.get(f"/api/target/{TID}").json()
    r = client.post("/api/evaluate", json={"target": TID, "torsions": d["native_torsions"]}).json()
    assert r["rmsd"] < 1e-3 and np.isfinite(r["terms"]["score"])
    assert r["terms"]["e_int"] == pytest.approx(r["terms"]["vdw"] + r["terms"]["elec"] + r["terms"]["solv"])
    assert client.post("/api/evaluate", json={"target": TID, "torsions": [0.0]}).status_code == 400
    assert client.get("/api/target/nope").status_code == 404


def test_randomize_is_seeded() -> None:
    a = client.get(f"/api/randomize/{TID}?seed=3").json()["torsions"]
    assert a == client.get(f"/api/randomize/{TID}?seed=3").json()["torsions"]


def test_search_job_runs_and_reports_history() -> None:
    jid = client.post("/api/job/start", json={"target": TID, "method": "hill_climb", "budget": 100, "seed": 0}).json()["id"]
    hist, s = [], {}
    for _ in range(120):
        s = client.get(f"/api/job/{jid}?since={len(hist)}").json()
        hist += s["new_history"]
        if s["status"] != "running":
            break
        time.sleep(0.5)
    assert s["status"] == "done" and s["calls"] == 100 and s["result"]["calls_used"] == 100
    assert len(hist) >= 1 and "V2000" in hist[0]["mol"] and hist[-1]["score"] == pytest.approx(s["best_score"])
    assert client.post("/api/job/start", json={"target": TID, "method": "bogus"}).status_code == 400


def test_results_report_and_policy_guards() -> None:
    r = client.get("/api/results").json()
    assert {"experiments", "tables", "figures", "aux"} <= set(r)
    assert client.get("/api/report").status_code == 200
    bad = client.post("/api/policy/act", json={"target": TID, "ckpt": "../../etc/passwd", "torsions": [0] * 8})
    assert bad.status_code == 400
    assert client.get("/").status_code == 200
