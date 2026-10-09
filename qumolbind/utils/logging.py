"""Run logging: CSV (always) + TensorBoard (optional)."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Append-only CSV logger with optional TensorBoard mirroring.

    The CSV header is fixed by the first row logged; later rows may add keys, in which case
    the file is rewritten with the widened header (rare, small files).
    """

    def __init__(self, run_dir: str | Path, name: str = "metrics", tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / f"{name}.csv"
        self._rows: list[dict[str, Any]] = []
        self._fields: list[str] = []
        self._t0 = time.time()
        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.run_dir / "tb"))
            except Exception:  # tensorboard optional
                self._tb = None

    def log(self, step: int, **metrics: float | int | str) -> None:
        row: dict[str, Any] = {"step": step, "wall_s": round(time.time() - self._t0, 3), **metrics}
        new_keys = [k for k in row if k not in self._fields]
        self._rows.append(row)
        if new_keys:
            self._fields += new_keys
            self._rewrite()
        else:
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writerow(row)
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)

    def _rewrite(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            w.writerows(self._rows)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
