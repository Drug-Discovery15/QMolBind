"""Config loading (Hydra compose API, so it works from scripts and tests alike)."""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(overrides: list[str] | None = None, config_name: str = "config") -> DictConfig:
    """Compose the Hydra config at ``configs/<config_name>.yaml`` with CLI-style overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    ov = list(overrides or [])
    if os.environ.get("QMB_TARGETS"):  # e.g. QMB_TARGETS=3ert: offline fallback to the committed fixture target (see Makefile/tasks.py)
        ov.append("experiment.targets=[" + os.environ["QMB_TARGETS"] + "]")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=ov)


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
