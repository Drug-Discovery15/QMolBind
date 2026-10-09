"""Config loading (Hydra compose API, so it works from scripts and tests alike)."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(overrides: list[str] | None = None, config_name: str = "config") -> DictConfig:
    """Compose the Hydra config at ``configs/<config_name>.yaml`` with CLI-style overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
