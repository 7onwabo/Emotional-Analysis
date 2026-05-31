"""Intermediate-task transfer (STILTs): fine-tune on auxiliary task then emotion.

Phase 1: fine-tune a transformer on AfriSenti (sentiment) or AfriHate (hate speech).
Phase 2: fine-tune the phase-1 checkpoint on BRIGHTER/EthioEmo emotion labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from emotion_analysis import EMOTION_LABELS
from emotion_analysis.data.datasets import build_split, resolve_languages
from emotion_analysis.data.loaders import (
    AFRISENTI_OVERLAP,
    AFRIHATE_OVERLAP,
    AuxExample,
    load_afrihate,
    load_afrisenti,
)
from emotion_analysis.models.transformer import build_hf_dataset, tokenize_function
from emotion_analysis.training.metrics import hf_compute_metrics_fn
from emotion_analysis.training.trainer import TrainArtifacts, train_transformer
from emotion_analysis.utils.config import load_config
from emotion_analysis.utils.seeds import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intermediate-task transfer training")
    p.add_argument("--aux", required=True, choices=["afrisenti", "afrihate"], help="Auxiliary dataset")
    p.add_argument("--model", default="afro_xlmr_base", help="Base transformer model key")
    p.add_argument("--aux-epochs", type=int, default=3, help="Epochs for auxiliary fine-tune")
    p.add_argument("--max-aux", type=int, default=5000, help="Max auxiliary examples per language (cap for memory/speed)")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def _aux_examples_to_dataset(
    examples: list[AuxExample], tokenizer: object, max_length: int, num_labels: int
) -> Dataset:
    texts = [ex.text for ex in examples]
    labels = [ex.label for ex in examples]
    ds = Dataset.from_dict({"text": texts, "labels": labels})
    ds = ds.map(
        lambda batch: tokenize_function(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    return ds


def fine_tune_auxiliary(
    hf_id: str,
    aux_examples_train: list[AuxExample],
    aux_examples_dev: list[AuxExample],
    num_labels: int,
    output_dir: Path,
    cfg: object,
    aux_epochs: int,
) -> tuple[object, object]:
    """Fine-tune a fresh transformer on the auxiliary task. Returns (model, tokenizer)."""
    import inspect

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=num_labels)

    max_length = 128
    train_ds = _aux_examples_to_dataset(aux_examples_train, tokenizer, max_length, num_labels)
    dev_ds = _aux_examples_to_dataset(aux_examples_dev, tokenizer, max_length, num_labels)

    t = cfg.train
    log = cfg.logging
    args = TrainingArguments(
        output_dir=str(output_dir / "aux_ckpt"),
        num_train_epochs=aux_epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=float(t.learning_rate),
        weight_decay=t.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=log.log_steps,
        report_to=[],
        seed=cfg.seed,
        dataloader_pin_memory=False,
        gradient_checkpointing=bool(getattr(t, "gradient_checkpointing", False)),
        gradient_checkpointing_kwargs={"use_reentrant": False} if getattr(t, "gradient_checkpointing", False) else None,
        use_cpu=True,  
    )
    if bool(getattr(t, "gradient_checkpointing", False)):
        model.config.use_cache = False

    trainer_kwargs: dict = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models", overrides=args.override)
    set_seed(cfg.seed)

    model_key = args.model
    spec = cfg.models[model_key]
    label_names = list(cfg.task.labels) or EMOTION_LABELS
    languages = resolve_languages(cfg, "all")

    aux_name = args.aux
    overlap = AFRISENTI_OVERLAP if aux_name == "afrisenti" else AFRIHATE_OVERLAP
    loader_fn = load_afrisenti if aux_name == "afrisenti" else load_afrihate
    num_aux_labels = 3 if aux_name == "afrisenti" else 2

    print(f"[transfer] aux={aux_name} model={model_key} overlap_langs={overlap}")

    aux_train: list[AuxExample] = []
    aux_dev: list[AuxExample] = []
    for lang in overlap:
        try:
            lang_train = loader_fn(lang, split="train")
            lang_dev = loader_fn(lang, split="dev" if aux_name == "afrisenti" else "validation")
            if args.max_aux and len(lang_train) > args.max_aux:
                import random; random.seed(42)
                lang_train = random.sample(lang_train, args.max_aux)
            aux_train += lang_train
            aux_dev += lang_dev[:500] 
            print(f"[transfer] loaded {lang} aux train={len(lang_train)} dev={len(lang_dev)}")
        except Exception as e:
            print(f"[transfer] warning: could not load {lang} from {aux_name}: {e}")

    if not aux_train:
        raise SystemExit(f"[transfer] No auxiliary data loaded for {aux_name}. Check HF hub access.")

    output_dir = Path(cfg.paths.output_dir) / f"{model_key}_{aux_name}_transfer_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[transfer] Phase 1: fine-tuning on {aux_name} ({len(aux_train)} examples)...")
    model, tokenizer = fine_tune_auxiliary(
        hf_id=spec.hf_id,
        aux_examples_train=aux_train,
        aux_examples_dev=aux_dev,
        num_labels=num_aux_labels,
        output_dir=output_dir,
        cfg=cfg,
        aux_epochs=args.aux_epochs,
    )

    print(f"[transfer] Phase 2: fine-tuning on emotion ({len(label_names)} labels)...")
    from transformers import AutoConfig

    emotion_config = AutoConfig.from_pretrained(
        spec.hf_id,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
    )
    emotion_model = AutoModelForSequenceClassification.from_config(emotion_config)
    encoder_state = {k: v for k, v in model.state_dict().items() if not k.startswith("classifier")}
    missing, unexpected = emotion_model.load_state_dict(encoder_state, strict=False)
    print(f"[transfer] encoder loaded — missing={len(missing)} unexpected={len(unexpected)}")

    train_texts, train_labels, _ = build_split(cfg, languages, "train")
    dev_texts, dev_labels, _ = build_split(cfg, languages, "dev")
    print(f"[transfer] emotion train={len(train_texts)} dev={len(dev_texts)}")

    train_ds = build_hf_dataset(train_texts, train_labels, tokenizer, spec.max_length)
    dev_ds = build_hf_dataset(dev_texts, dev_labels, tokenizer, spec.max_length)

    arr = np.asarray(train_labels, dtype="float32")
    pos_weight = ((len(arr) - arr.sum(0)) / np.maximum(arr.sum(0), 1)).tolist()

    artifacts = train_transformer(
        emotion_model,
        tokenizer,
        train_ds,
        dev_ds,
        config=cfg,
        output_dir=output_dir,
        label_names=label_names,
        pos_weight=pos_weight,
    )

    (output_dir / "dev_metrics.json").write_text(json.dumps(artifacts.metrics, indent=2))
    (output_dir / "run_config.json").write_text(json.dumps({
        "model_key": model_key,
        "aux_dataset": aux_name,
        "languages": languages,
        "label_names": label_names,
        "aux_train_size": len(aux_train),
        "emotion_train_size": len(train_texts),
    }, indent=2))
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")
    print(f"[transfer] dev f1_macro={artifacts.metrics.get('f1_macro'):.4f}")
    print(f"[transfer] artifacts -> {output_dir}")


if __name__ == "__main__":
    main()
