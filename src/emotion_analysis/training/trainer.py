"""Training orchestration.

Wraps `transformers.Trainer` for the transformer path; for sklearn baselines
runs a straight `fit()` loop. Both paths emit the same metrics dict (keyed by
`f1_macro`, per-label `f1_*`, etc.) so the evaluation code is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emotion_analysis import EMOTION_LABELS
from emotion_analysis.training.metrics import (
    hf_compute_metrics_fn,
    metrics_from_predictions,
)


@dataclass
class TrainArtifacts:
    model: Any
    tokenizer: Any | None
    metrics: dict[str, float]
    output_dir: Path


def _freeze_base_layers(model: Any, n: int) -> None:
    """Freeze embeddings + the bottom `n` encoder layers of an XLM-R/BERT model.

    Cuts gradient + optimizer-state memory (only trainable params get Adam state).
    Best-effort across HF encoder naming; no-op if the structure isn't found.
    """
    base = getattr(model, "base_model", model)
    embeddings = getattr(base, "embeddings", None)
    if embeddings is not None:
        for p in embeddings.parameters():
            p.requires_grad_(False)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None) if encoder is not None else None
    if layers is not None:
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad_(False)


def train_transformer(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    config: Any,
    output_dir: str | Path,
    label_names: list[str] | None = None,
    pos_weight: list[float] | None = None,
) -> TrainArtifacts:
    """Fine-tune via HuggingFace Trainer.

    `config` is the merged OmegaConf with `train`, `logging`, `task` sections.
    `train_ds` / `eval_ds` are tokenized datasets with float `labels` (see
    `models.transformer.build_hf_dataset`).
    """
    from transformers import (
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    label_names = label_names or list(EMOTION_LABELS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t = config.train
    log = config.logging
    threshold = float(config.task.threshold)

    freeze_n = int(getattr(t, "freeze_base_layers", 0) or 0)
    if freeze_n > 0:
        _freeze_base_layers(model, freeze_n)

    gradient_checkpointing = bool(getattr(t, "gradient_checkpointing", False))
    if gradient_checkpointing:
        model.config.use_cache = False  

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t.num_epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=float(t.learning_rate),
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler,
        max_grad_norm=t.max_grad_norm,
        fp16=t.fp16,
        bf16=t.bf16,
        eval_strategy=log.eval_strategy,
        save_strategy=log.save_strategy,
        save_total_limit=log.save_total_limit,
        logging_steps=log.log_steps,
        load_best_model_at_end=True,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        report_to=list(log.report_to),
        seed=config.seed,
        dataloader_pin_memory=False, 
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
    )

    import inspect

    kwargs: dict[str, Any] = {"pos_weight": pos_weight}
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_compute_metrics_fn(label_names, threshold=threshold),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t.early_stopping_patience)],
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    pos_weight = kwargs.get("pos_weight", None)
    if pos_weight is not None:
        import torch

        class _WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **_kw):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
                loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer = _WeightedTrainer(**trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return TrainArtifacts(
        model=model,
        tokenizer=tokenizer,
        metrics={k.replace("eval_", ""): v for k, v in metrics.items()},
        output_dir=output_dir,
    )


def train_baseline(
    pipeline: Any,
    train_texts: list[str],
    train_labels: Any,
    eval_texts: list[str],
    eval_labels: Any,
    output_dir: str | Path,
    label_names: list[str] | None = None,
) -> TrainArtifacts:
    """Fit the sklearn pipeline and evaluate on the dev split.

    Persists the fitted pipeline to `output_dir/pipeline.joblib`.
    """
    import joblib
    import numpy as np

    label_names = label_names or list(EMOTION_LABELS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline.fit(train_texts, np.asarray(train_labels))
    preds = np.asarray(pipeline.predict(eval_texts))
    metrics = metrics_from_predictions(preds, np.asarray(eval_labels), label_names=label_names)

    joblib.dump(pipeline, output_dir / "pipeline.joblib")

    return TrainArtifacts(
        model=pipeline,
        tokenizer=None,
        metrics=metrics,
        output_dir=output_dir,
    )
