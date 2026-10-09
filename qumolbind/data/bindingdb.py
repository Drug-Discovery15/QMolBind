"""Per-target BindingDB affinities (REST, by UniProt id). Never the full dump."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger

BINDINGDB_URL = "https://www.bindingdb.org/rest/getLigandsByUniprots"
TYPE_PRIORITY = {"Kd": 0, "Ki": 1, "IC50": 2}  # prefer direct binding constants when deduping

RDLogger.DisableLog("rdApp.*")


def fetch_raw(uniprot: str, cutoff_nm: float = 100000, timeout: float = 180, retries: int = 4) -> list[dict]:
    """GET with exponential backoff (the public endpoint returns transient 503s)."""
    import time

    last: Exception | None = None
    for k in range(retries):
        try:
            r = requests.get(BINDINGDB_URL, params={"uniprot": uniprot, "cutoff": cutoff_nm, "response": "application/json"}, timeout=timeout)
            r.raise_for_status()
            return r.json()["getLindsByUniprotsResponse"]["affinities"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * 2**k)
    raise RuntimeError(f"BindingDB unavailable for {uniprot}: {last}")


def p_affinity(value_nm: float) -> float:
    """pAffinity = -log10(molar) = 9 - log10(nM)."""
    return 9.0 - float(np.log10(value_nm))


def normalize(records: list[dict]) -> pd.DataFrame:
    """Raw records -> one row per canonical SMILES. Censored (<, >) and non-Ki/Kd/IC50 values are dropped."""
    rows = []
    for rec in records:
        t = rec.get("affinity_type")
        v = str(rec.get("affinity", "")).strip()
        if t not in TYPE_PRIORITY or not v or v[0] in "<>~":
            continue
        try:
            nm = float(v)
        except ValueError:
            continue
        mol = Chem.MolFromSmiles(rec.get("smile", ""))
        if nm <= 0 or mol is None:
            continue
        rows.append(
            {"smiles": Chem.MolToSmiles(mol), "measurement": t, "p_affinity": p_affinity(nm), "monomerid": rec.get("monomerid")}
        )
    if not rows:
        return pd.DataFrame(columns=["smiles", "measurement", "p_affinity", "n_measurements", "monomerid"])
    df = pd.DataFrame(rows)
    df["prio"] = df["measurement"].map(TYPE_PRIORITY)
    out = []
    for smi, g in df.groupby("smiles", sort=True):
        best = g[g["prio"] == g["prio"].min()]  # keep only the most preferred measurement type
        out.append(
            {
                "smiles": smi,
                "measurement": best["measurement"].iloc[0],
                "p_affinity": float(best["p_affinity"].median()),
                "n_measurements": len(g),
                "monomerid": best["monomerid"].iloc[0],
            }
        )
    return pd.DataFrame(out)


def load_target(uniprot: str, cache_dir: str | Path, fixture: str | Path | None = None) -> pd.DataFrame:
    """Fetch (or read from cache) and normalize affinities for one target.

    ``fixture`` must be a slice of THIS uniprot's data (offline fallback); never pass another target's file."""
    cache = Path(cache_dir) / f"bindingdb_{uniprot}.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
    else:
        try:
            recs = fetch_raw(uniprot)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(recs))
        except Exception as e:
            if fixture is None:
                raise
            print(f"[bindingdb] network failure ({e}); using fixture {fixture}")
            recs = json.loads(Path(fixture).read_text())
    return normalize(recs)
