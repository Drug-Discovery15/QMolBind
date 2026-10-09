"""Generate REPORT.md from results/ and print which branch of the pre-registered criterion applied (mechanically, from the data)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qumolbind.eval.report import build_report, update_architecture  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smoke")
    ap.add_argument("--out", default=str(ROOT / "REPORT.md"))
    a = ap.parse_args()
    text, verdict = build_report(a.experiment)
    update_architecture()
    Path(a.out).write_text(text, encoding="utf-8")
    (ROOT / "results" / f"verdict_{a.experiment}.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(f"wrote {a.out}")
    print(f"PRE-REGISTERED CRITERION -> {verdict['branch']}")
    for c in verdict["failed_conditions"]:
        print(f"  - {c}")
