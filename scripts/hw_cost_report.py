"""Transpile encoding + ansatz to a Heron-like basis (cz/rz/sx/x) + heavy-hex coupling; log 2-qubit gate count and depth."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.quantum.hw_cost import hardware_cost_report  # noqa: E402

if __name__ == "__main__":
    rows = []
    for ent in ("cnot", "cz"):
        for r in hardware_cost_report(out_csv=None, entangler=ent):
            r["entangler"] = ent
            rows.append(r)
    import pandas as pd

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "hw_cost.csv", index=False)
    print(df[["entangler", "n_qubits", "amplitude_dim", "cz_encoding_plus_ansatz", "depth_encoding_plus_ansatz", "cz_ansatz_only", "depth_ansatz_only"]].to_string(index=False))
