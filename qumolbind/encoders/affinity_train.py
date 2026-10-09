"""Affinity-model training on pilot BindingDB data: GNN vs LightGBM-on-ECFP4, scaffold split, test RMSE.

The affinity model is NOT the RL reward; it only supplies a (frozen, cached) ligand embedding and an optional surrogate.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.loader import DataLoader

from qumolbind.encoders.gnn import AffinityGNN, smiles_to_data

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"


def scaffold_split(smiles: list[str], frac: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0):
    """Bemis-Murcko scaffold split: whole scaffold groups go to one side (largest groups -> train)."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
        except Exception:
            sc = ""
        groups.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    big = sorted(groups.values(), key=lambda g: (-len(g), rng.random()))
    n = len(smiles)
    tr, va, te = [], [], []
    for g in big:
        if len(tr) + len(g) <= frac[0] * n:
            tr += g
        elif len(va) + len(g) <= frac[1] * n:
            va += g
        else:
            te += g
    return np.array(tr), np.array(va), np.array(te)


def ecfp4(smiles: list[str], n_bits: int = 2048) -> np.ndarray:
    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=n_bits)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), X[i])
    return X


def _feat(args):
    smi, y, seed = args
    return smiles_to_data(smi, y, seed=seed)


def build_graphs(df: pd.DataFrame, cache_path: Path, workers: int = 12, seed: int = 0) -> list:
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    jobs = [(s, float(y), seed) for s, y in zip(df.smiles, df.p_affinity)]
    with mp.get_context("spawn").Pool(workers) as p:
        out = p.map(_feat, jobs, chunksize=32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def evaluate(model: AffinityGNN, graphs: list, bs: int = 64) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in DataLoader(graphs, batch_size=bs):
            preds.append(model(b).numpy()); ys.append(b.y.numpy())
    return np.concatenate(preds), np.concatenate(ys)


def run_target(
    target_id: str, kind: str = "dimenet", max_molecules: int = 2500, epochs: int = 25, lr: float = 1e-3,
    seed: int = 0, hidden: int = 64, out_dir: Path | None = None,
) -> dict:
    import h5py

    from qumolbind.utils.logging import RunLogger
    from qumolbind.utils.seeding import seed_everything

    seed_everything(seed)
    out_dir = out_dir or ROOT / "results" / "affinity" / target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(CACHE / "processed" / f"{target_id}.h5") as f:
        g = f["affinity"]
        df = pd.DataFrame({"smiles": [s.decode() for s in g["smiles"][:]], "p_affinity": g["p_affinity"][:]})
    if len(df) > max_molecules:
        df = df.sample(max_molecules, random_state=seed).reset_index(drop=True)
    graphs = build_graphs(df, CACHE / "affinity" / f"{target_id}_{len(df)}_graphs.pt", seed=seed)
    keep = [i for i, d in enumerate(graphs) if d is not None]
    failed = len(graphs) - len(keep)
    df, graphs = df.iloc[keep].reset_index(drop=True), [graphs[i] for i in keep]
    tr, va, te = scaffold_split(list(df.smiles), seed=seed)
    ymu, ysd = float(df.p_affinity.iloc[tr].mean()), float(df.p_affinity.iloc[tr].std() + 1e-8)

    # --- LightGBM on ECFP4 baseline
    import lightgbm as lgb

    X = ecfp4(list(df.smiles))
    y = df.p_affinity.to_numpy()
    gbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    gbm.fit(X[tr], y[tr])
    lgb_rmse = rmse(gbm.predict(X[te]), y[te])
    mean_rmse = rmse(np.full(len(te), y[tr].mean()), y[te])

    # --- GNN
    for d in graphs:
        d.y = (d.y - ymu) / ysd
    model = AffinityGNN(kind, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lg = RunLogger(out_dir, f"train_{kind}", tensorboard=False)
    best_val, best_state = np.inf, None
    t0 = time.time()
    loader = DataLoader([graphs[i] for i in tr], batch_size=32, shuffle=True)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for b in loader:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(b), b.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * b.num_graphs
        p, t = evaluate(model, [graphs[i] for i in va])
        v = rmse(p * ysd, t * ysd)
        lg.log(ep, train_mse=tot / len(tr), val_rmse=v)
        if v < best_val:
            best_val, best_state = v, {k: x.clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    p, t = evaluate(model, [graphs[i] for i in te])
    gnn_rmse = rmse(p * ysd, t * ysd)

    # frozen embedding statistics (used to normalise the RL ligand embedding)
    model.eval()
    with torch.no_grad():
        embs = torch.cat([torch.nn.functional.silu(model.embed(b)) for b in DataLoader([graphs[i] for i in tr], batch_size=64)]).numpy()
    enc_dir = CACHE / "encoders"
    enc_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "kind": kind, "hidden": hidden, "emb_mean": embs.mean(0), "emb_std": embs.std(0) + 1e-6}, enc_dir / f"gnn_{target_id}.pt")
    metrics = {
        "target": target_id, "kind": kind, "n_total": len(df), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "featurisation_failures": failed, "epochs": epochs, "seed": seed,
        "test_rmse_gnn": gnn_rmse, "test_rmse_lightgbm_ecfp4": lgb_rmse, "test_rmse_predict_train_mean": mean_rmse,
        "best_val_rmse_gnn": best_val, "train_seconds": time.time() - t0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def table_from_logs(root: Path = ROOT / "results" / "affinity") -> str:
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/metrics.json"))]
    L = ["| target | GNN | n train/val/test | GNN test RMSE | LightGBM-ECFP4 test RMSE | train-mean RMSE |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['target']} | {r['kind']} | {r['n_train']}/{r['n_val']}/{r['n_test']} | {r['test_rmse_gnn']:.3f} | "
                 f"{r['test_rmse_lightgbm_ecfp4']:.3f} | {r['test_rmse_predict_train_mean']:.3f} |")
    return "\n".join(L)
