"""Optional ZINC-22 tranche loader (stretch only; not used by the pose-search MVP).

Access method (checked at implementation time): ZINC-22 tranches are served from
https://files.docking.org/zinc22/2d-<tranche>/... as ``.smi`` (SMILES + zinc id) or gzipped ``.smi.gz``.
Pass a URL or a local path. The default behaviour never downloads anything.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd
import requests


def load_tranche(source: str | Path, max_molecules: int = 50_000, timeout: float = 120) -> pd.DataFrame:
    """Load <= max_molecules rows (smiles, zinc_id) from a local file or http(s) URL (.smi or .smi.gz)."""
    src = str(source)
    if src.startswith(("http://", "https://")):
        r = requests.get(src, timeout=timeout, stream=True)
        r.raise_for_status()
        raw = r.content
    else:
        raw = Path(src).read_bytes()
    if src.endswith(".gz"):
        raw = gzip.decompress(raw)
    rows = []
    for line in io.StringIO(raw.decode()):
        parts = line.split()
        if len(parts) >= 1 and parts[0].lower() != "smiles":
            rows.append({"smiles": parts[0], "zinc_id": parts[1] if len(parts) > 1 else ""})
        if len(rows) >= max_molecules:
            break
    return pd.DataFrame(rows)
