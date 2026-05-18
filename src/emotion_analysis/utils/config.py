"""Config loading via OmegaConf.

Merges base configs in `configs/` and applies CLI overrides.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_config(
    *names: str,
    overrides: list[str] | None = None,
    config_dir: Path | None = None,
) -> DictConfig:
    """Load and merge YAML configs.

    Usage:
        cfg = load_config("training", "data", "languages", overrides=["train.batch_size=8"])
    """
    config_dir = config_dir or CONFIG_DIR
    parts = [OmegaConf.load(config_dir / f"{name}.yaml") for name in names]
    cfg = OmegaConf.merge(*parts)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]
