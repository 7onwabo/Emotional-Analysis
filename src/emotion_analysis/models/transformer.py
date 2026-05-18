"""Transformer fine-tuning wrapper around HuggingFace `Trainer`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransformerConfig:
    hf_id: str
    num_labels: int
    max_length: int = 128
    problem_type: str = "multi_label_classification"
    label_names: list[str] = field(default_factory=list)


def load_model_and_tokenizer(config: TransformerConfig) -> tuple[Any, Any]:
    """Load `AutoModelForSequenceClassification` + `AutoTokenizer`.

    Returns (model, tokenizer).
    """
    raise NotImplementedError("TODO: AutoModelForSequenceClassification.from_pretrained")


def tokenize_function(examples: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    raise NotImplementedError("TODO: tokenizer(examples['text'], truncation=True, max_length=max_length)")
