"""Training orchestration.

Wraps `transformers.Trainer` for the transformer path; for sklearn baselines
runs a straight `fit()` loop. Both paths emit the same metrics dict so the
evaluation code is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainArtifacts:
    model: Any
    tokenizer: Any | None
    metrics: dict[str, float]
    output_dir: Path


def train_transformer(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    config: Any,
) -> TrainArtifacts:
    """Fine-tune via HuggingFace Trainer.

    Expected to:
        1. Build `TrainingArguments` from config.train + config.logging.
        2. Build `Trainer(model, args, train_ds, eval_ds, compute_metrics, ...)`.
        3. Call `trainer.train()` and `trainer.evaluate()`.
    """
    raise NotImplementedError("TODO: HF Trainer wiring")


def train_baseline(
    pipeline: Any,
    train_texts: list[str],
    train_labels: Any,
    eval_texts: list[str],
    eval_labels: Any,
    config: Any,
) -> TrainArtifacts:
    raise NotImplementedError("TODO: pipeline.fit + predict + metrics")
