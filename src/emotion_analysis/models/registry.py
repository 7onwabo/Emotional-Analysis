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
    if spec.type == "bilstm":
        from emotion_analysis.models.bilstm import BiLSTMClassifier, BiLSTMConfig, FastTextTokenizer

        config = BiLSTMConfig(
            num_labels=num_labels,
            vocab_size=getattr(spec, "vocab_size", 200_000),
            embed_dim=getattr(spec, "embed_dim", 300),
            hidden_dim=getattr(spec, "hidden_dim", 256),
            num_layers=getattr(spec, "num_layers", 2),
            dropout=getattr(spec, "dropout", 0.3),
            max_length=getattr(spec, "max_length", 128),
        )
        return BiLSTMClassifier(config), FastTextTokenizer(
            vocab_size=config.vocab_size, max_length=config.max_length
        )
    raise ValueError(f"Unknown model type: {spec.type}")
