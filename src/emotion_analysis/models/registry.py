"""Model factory: dispatch on config `type` field."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def list_models(config_path: str | Path = "configs/models.yaml") -> list[str]:
    cfg = OmegaConf.load(config_path)
    return list(cfg.models.keys())


def build_model(model_key: str, num_labels: int, config_path: str | Path = "configs/models.yaml") -> Any:
    cfg = OmegaConf.load(config_path)
    if model_key not in cfg.models:
        raise KeyError(f"Unknown model '{model_key}'. Available: {list(cfg.models.keys())}")

    spec = cfg.models[model_key]
    if spec.type == "sklearn":
        from emotion_analysis.models.baseline import BaselineConfig, build_baseline

        return build_baseline(
            BaselineConfig(
                vectorizer=spec.vectorizer,
                classifier=spec.classifier,
                **OmegaConf.to_container(spec.params, resolve=True),
            )
        )
    if spec.type == "transformer":
        from emotion_analysis.models.transformer import TransformerConfig, load_model_and_tokenizer

        return load_model_and_tokenizer(
            TransformerConfig(
                hf_id=spec.hf_id,
                num_labels=num_labels,
                max_length=spec.max_length,
            )
        )
    raise ValueError(f"Unknown model type: {spec.type}")
