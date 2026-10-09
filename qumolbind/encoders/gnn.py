"""Ligand GNN (configurable dimenet | schnet | gine). Produces a 64-d graph embedding + an affinity head.

pyg-lib / torch-cluster / torch-sparse have no wheels for this torch build, so ``patch_pyg`` swaps PyG's
``radius_graph`` and DimeNet ``triplets`` for pure-torch equivalents (tested against brute force; D15 in DECISIONS.md).
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch import nn
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")
EMB_DIM = 64
BOND_TYPES = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
_PATCHED = False


# ---------------------------------------------------------------------------- pure-torch PyG replacements
def radius_graph_torch(x: torch.Tensor, r: float, batch: torch.Tensor | None = None, loop: bool = False,
                       max_num_neighbors: int = 32, flow: str = "source_to_target", **_) -> torch.Tensor:
    n = x.size(0)
    batch = torch.zeros(n, dtype=torch.long, device=x.device) if batch is None else batch
    d = torch.cdist(x, x)
    mask = (d <= r) & (batch[:, None] == batch[None, :])
    if not loop:
        mask &= ~torch.eye(n, dtype=torch.bool, device=x.device)
    if max_num_neighbors is not None and mask.sum(1).max() > max_num_neighbors:  # keep nearest neighbours per centre
        dd = torch.where(mask, d, torch.full_like(d, float("inf")))
        kth = dd.topk(max_num_neighbors, dim=1, largest=False).values[:, -1:]
        mask &= dd <= kth
    centre, nbr = mask.nonzero(as_tuple=True)  # row = centre i, col = neighbour j
    return torch.stack([nbr, centre], 0) if flow == "source_to_target" else torch.stack([centre, nbr], 0)


def triplets_torch(edge_index: torch.Tensor, num_nodes: int):
    """Same outputs as torch_geometric.nn.models.dimenet.triplets without torch_sparse."""
    row, col = edge_index  # edge e: j=row[e] -> i=col[e]
    E = row.numel()
    order = torch.argsort(col)
    counts = torch.bincount(col, minlength=num_nodes)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    n_trip = counts[row]                                   # edges (k -> j) entering j, for each edge (j -> i)
    idx_ji = torch.arange(E, device=row.device).repeat_interleave(n_trip)
    starts = ptr[row].repeat_interleave(n_trip)
    offs = torch.arange(idx_ji.numel(), device=row.device) - (n_trip.cumsum(0) - n_trip).repeat_interleave(n_trip)
    idx_kj = order[starts + offs]
    idx_i, idx_j, idx_k = col[idx_ji], row[idx_ji], row[idx_kj]
    keep = idx_i != idx_k
    return col, row, idx_i[keep], idx_j[keep], idx_k[keep], idx_kj[keep], idx_ji[keep]


def patch_pyg() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import torch_geometric.nn.models.dimenet as dn
    import torch_geometric.nn.models.schnet as sn

    dn.radius_graph = radius_graph_torch
    dn.triplets = triplets_torch
    sn.radius_graph = radius_graph_torch
    _PATCHED = True


# ---------------------------------------------------------------------------- featurisation
def smiles_to_data(smiles: str, y: float | None = None, n_conf: int = 5, seed: int = 0) -> Data | None:
    """ETKDGv3 conformers -> lowest MMFF energy; heavy-atom graph with 3D positions. None if embedding fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.useRandomCoords = True
    ids = list(AllChem.EmbedMultipleConfs(mol, n_conf, ps))
    if not ids:
        return None
    best, best_e = ids[0], np.inf
    if AllChem.MMFFHasAllMoleculeParams(mol):
        for cid, (conv, e) in zip(ids, AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)):
            if e < best_e:
                best, best_e = cid, e
    heavy = Chem.RemoveHs(mol)
    pos = torch.tensor(mol.GetConformer(best).GetPositions()[: heavy.GetNumAtoms()], dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in heavy.GetAtoms()], dtype=torch.long)
    ei, ea = [], []
    for b in heavy.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = BOND_TYPES.get(b.GetBondType(), 0)
        ei += [[i, j], [j, i]]; ea += [t, t]
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros(2, 0, dtype=torch.long)
    d = Data(z=z, pos=pos, edge_index=edge_index, edge_attr=torch.tensor(ea, dtype=torch.long))
    d.smiles = smiles
    if y is not None:
        d.y = torch.tensor([y], dtype=torch.float32)
    return d


# ---------------------------------------------------------------------------- model
class GINEBackbone(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 3, out: int = EMB_DIM) -> None:
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.emb = nn.Embedding(100, hidden)
        self.bond_emb = nn.ModuleList([nn.Embedding(4, hidden) for _ in range(layers)])
        self.convs = nn.ModuleList(
            [GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))) for _ in range(layers)]
        )
        self.out = nn.Linear(hidden, out)

    def forward(self, z, pos, batch, edge_index, edge_attr):
        from torch_geometric.nn import global_add_pool

        h = self.emb(z)
        for conv, be in zip(self.convs, self.bond_emb):
            h = torch.relu(conv(h, edge_index, be(edge_attr))) + h
        return self.out(global_add_pool(h, batch))


class AffinityGNN(nn.Module):
    """backbone -> 64-d embedding -> SiLU -> Linear -> pAffinity (SiLU: DimeNet zero-inits its output layer, so ReLU(0) would be a dead unit)."""

    def __init__(self, kind: str = "dimenet", hidden: int = 64) -> None:
        super().__init__()
        patch_pyg()
        self.kind = kind
        if kind == "dimenet":
            from torch_geometric.nn import DimeNetPlusPlus

            self.backbone = DimeNetPlusPlus(
                hidden_channels=hidden, out_channels=EMB_DIM, num_blocks=2, int_emb_size=32, basis_emb_size=8,
                out_emb_channels=hidden, num_spherical=3, num_radial=4, cutoff=5.0, num_output_layers=2,
            )
        elif kind == "schnet":
            from torch_geometric.nn import SchNet

            self.backbone = SchNet(hidden_channels=hidden, num_filters=hidden, num_interactions=3, cutoff=5.0)
            self.backbone.lin2 = nn.Linear(hidden // 2, EMB_DIM)  # per-atom 64-d, summed by the readout
        elif kind == "gine":
            self.backbone = GINEBackbone(hidden)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(EMB_DIM, 1)

    def embed(self, b) -> torch.Tensor:
        if self.kind == "gine":
            out = self.backbone(b.z, b.pos, b.batch, b.edge_index, b.edge_attr)
        else:
            out = self.backbone(b.z, b.pos, b.batch)
        return out.reshape(-1, EMB_DIM)

    def forward(self, b) -> torch.Tensor:
        return self.head(torch.nn.functional.silu(self.embed(b))).squeeze(-1)


def ligand_embedding(target_id: str, smiles: str, cache_dir=None) -> np.ndarray:
    """Frozen, cached 64-d ligand embedding of the target's own ligand (ETKDG conformer, NOT the crystal pose).

    Standardised with the training-set statistics of the affinity model, clipped to +-3 and divided by 3.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "data_cache"
    cache = Path(cache_dir or root / "embeddings") / f"{target_id}_ligand.npy"
    if cache.exists():
        return np.load(cache)
    ck = torch.load(root / "encoders" / f"gnn_{target_id}.pt", weights_only=False)
    model = AffinityGNN(ck["kind"], ck["hidden"])
    model.load_state_dict(ck["state"])
    model.eval()
    from torch_geometric.data import Batch

    b = Batch.from_data_list([smiles_to_data(smiles)])
    with torch.no_grad():
        e = torch.nn.functional.silu(model.embed(b))[0].numpy()
    e = np.clip((e - ck["emb_mean"]) / ck["emb_std"], -3, 3) / 3.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, e.astype(np.float32))
    return e.astype(np.float32)
