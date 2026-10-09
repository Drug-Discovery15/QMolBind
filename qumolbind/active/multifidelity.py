"""Multi-fidelity record keeping. Every number is tagged with the fidelity that produced it:
  L0 = surrogate ensemble (predicted), L1 = fast oracle (static pose, OpenMM GBn2), L2 = slow oracle (MD MM-GBSA-style).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd


@dataclass
class Record:
    round: int
    cand_id: int
    features: np.ndarray = field(repr=False)
    l1_score: float = np.nan          # fidelity L1 (kJ/mol; E_int + strain)
    l1_e_int: float = np.nan
    l0_mu_dg: float = np.nan          # fidelity L0 prediction of the L2 value (kJ/mol), made BEFORE the L2 label existed
    l0_sigma_dg: float = np.nan
    l0_mu_t: float = np.nan           # same in symlog target units
    l0_sigma_t: float = np.nan
    selected_for_l2: bool = False
    l2_dg: float = np.nan             # fidelity L2 (kJ/mol)
    l2_sem: float = np.nan
    l2_seconds: float = np.nan
    l2_failed: bool = False
    acquisition: float = np.nan


class MultiFidelityDataset:
    def __init__(self) -> None:
        self.records: list[Record] = []

    def add(self, r: Record) -> Record:
        self.records.append(r)
        return r

    def labelled(self) -> list[Record]:
        return [r for r in self.records if r.selected_for_l2 and not r.l2_failed and np.isfinite(r.l2_dg)]

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        L = self.labelled()
        if not L:
            return np.zeros((0, len(self.records[0].features) if self.records else 0)), np.zeros(0)
        return np.stack([r.features for r in L]), np.array([r.l2_dg for r in L])

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            d = {k: v for k, v in asdict(r).items() if k != "features"}
            d["fidelity_l1"], d["fidelity_l0"], d["fidelity_l2"] = "L1:fast_oracle", "L0:surrogate", "L2:md_mmgbsa" if r.selected_for_l2 else ""
            rows.append(d)
        return pd.DataFrame(rows)
