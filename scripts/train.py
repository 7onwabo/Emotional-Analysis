"""Train a model on the configured language(s).

Examples:
    python scripts/train.py --model tfidf_logreg --language afr
    python scripts/train.py --model afro_xlmr_base --language all
    python scripts/train.py --model afro_xlmr_base --language swa --override train.num_epochs=3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

from emotion_analysis import EMOTION_LABELS  # noqa: E402
from emotion_analysis.data.datasets import build_split, resolve_languages  # noqa: E402
from emotion_analysis.models.registry import build_model  # noqa: E402
from emotion_analysis.utils.config import load_config  # noqa: E402
from emotion_analysis.utils.seeds import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train emotion classifier")
    p.add_argument("--model", default=None, help="Model key from configs/models.yaml")
    p.add_argument("--language", default="all", help="Target ISO code, or 'all' for every target")
    p.add_argument("--override", nargs="*", default=[], help="Dotlist overrides, e.g. train.batch_size=8")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models", overrides=args.override)
    set_seed(cfg.seed)

    model_key = args.model or cfg.default
    spec = cfg.models[model_key]
    languages = resolve_languages(cfg, args.language)
    label_names = list(cfg.task.labels) or EMOTION_LABELS
    lang_tag = "all" if len(languages) > 1 else languages[0]

    print(f"[train] model={model_key} type={spec.type} languages={languages} seed={cfg.seed}")

    train_texts, train_labels, _ = build_split(cfg, languages, "train")
    dev_texts, dev_labels, _ = build_split(cfg, languages, "dev")
    print(f"[train] train={len(train_texts)} dev={len(dev_texts)}")

    output_dir = Path(cfg.paths.output_dir) / f"{model_key}_{lang_tag}"

    if spec.type == "sklearn":
        from emotion_analysis.training.trainer import train_baseline

        pipeline = build_model(model_key, num_labels=len(label_names))
        artifacts = train_baseline(
            pipeline,
            list(train_texts),
            train_labels,
            list(dev_texts),
            dev_labels,
            output_dir=output_dir,
            label_names=label_names,
        )
    elif spec.type == "transformer":
        from emotion_analysis.models.transformer import build_hf_dataset
        from emotion_analysis.training.trainer import train_transformer

        model, tokenizer = build_model(model_key, num_labels=len(label_names))
        train_ds = build_hf_dataset(train_texts, train_labels, tokenizer, spec.max_length)
        dev_ds = build_hf_dataset(dev_texts, dev_labels, tokenizer, spec.max_length)
        artifacts = train_transformer(
            model,
            tokenizer,
            train_ds,
            dev_ds,
            config=cfg,
            output_dir=output_dir,
            label_names=label_names,
        )
    else:
        raise SystemExit(f"Unknown model type: {spec.type}")

    (artifacts.output_dir / "dev_metrics.json").write_text(json.dumps(artifacts.metrics, indent=2))
    run_meta = {
        "model_key": model_key,
        "model_type": spec.type,
        "languages": languages,
        "label_names": label_names,
        "n_train": len(train_texts),
        "n_dev": len(dev_texts),
    }
    (artifacts.output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2))
    OmegaConf.save(cfg, artifacts.output_dir / "resolved_config.yaml")

    print(f"[train] dev f1_macro={artifacts.metrics.get('f1_macro'):.4f}")
    print(f"[train] artifacts -> {artifacts.output_dir}")


if __name__ == "__main__":
    main()
