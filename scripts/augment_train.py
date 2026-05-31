"""Back-translation augmentation + retrain (RQ3).
Augments training data for specified languages via Google Translate round-trip,
then retrains the best model (AfroXLMR) on the augmented set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from omegaconf import OmegaConf

from emotion_analysis import EMOTION_LABELS
from emotion_analysis.data.augmentation import BackTranslationAugmenter
from emotion_analysis.data.datasets import build_split, resolve_languages
from emotion_analysis.data.loaders import EmotionExample, to_arrays
from emotion_analysis.models.registry import build_model
from emotion_analysis.models.transformer import build_hf_dataset
from emotion_analysis.training.trainer import train_transformer
from emotion_analysis.utils.config import load_config
from emotion_analysis.utils.seeds import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment training data and retrain")
    p.add_argument("--languages", nargs="+", default=["afr"], help="Languages to augment")
    p.add_argument("--model", default="afro_xlmr_base")
    p.add_argument("--max-per-lang", type=int, default=300, help="Max augmented examples per language")
    p.add_argument("--cache", default="data/augmented_cache.json", help="Cache file to avoid re-translating")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def load_cache(path: str) -> dict[str, str]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_cache(path: str, cache: dict[str, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models", overrides=args.override)
    set_seed(cfg.seed)

    model_key = args.model
    spec = cfg.models[model_key]
    label_names = list(cfg.task.labels) or EMOTION_LABELS
    all_languages = resolve_languages(cfg, "all")

    aug_langs = args.languages
    for lang in aug_langs:
        if lang not in all_languages:
            raise SystemExit(f"Unknown language '{lang}'. Available: {all_languages}")

    cache = load_cache(args.cache)
    augmenter = BackTranslationAugmenter(delay=0.3)

    train_texts, train_labels, train_langs_list = build_split(cfg, all_languages, "train")
    dev_texts, dev_labels, _ = build_split(cfg, all_languages, "dev")

    aug_texts: list[str] = []
    aug_labels: list[list[int]] = []

    for lang in aug_langs:
        lang_indices = [i for i, l in enumerate(train_langs_list) if l == lang]
        n_aug = min(args.max_per_lang, len(lang_indices))
        rng = np.random.default_rng(cfg.seed)
        chosen = rng.choice(lang_indices, size=n_aug, replace=False)
        print(f"[augment] {lang}: augmenting {n_aug} examples via back-translation...")

        done = 0
        for idx in chosen:
            text = train_texts[idx]
            cache_key = f"{lang}:{text[:80]}"
            if cache_key in cache:
                aug = cache[cache_key]
            else:
                results = augmenter.augment(text, lang)
                aug = results[0] if results else ""
                cache[cache_key] = aug
                if done % 50 == 0:
                    save_cache(args.cache, cache)
            if aug:
                aug_texts.append(aug)
                aug_labels.append(list(train_labels[idx]))
            done += 1
            if done % 100 == 0:
                print(f"[augment]   {lang}: {done}/{n_aug} done, {len(aug_texts)} augmented so far")

        save_cache(args.cache, cache)

    print(f"[augment] total augmented examples: {len(aug_texts)}")

    combined_texts = list(train_texts) + aug_texts
    combined_labels = np.vstack([train_labels, np.array(aug_labels, dtype=np.int8)]) if aug_labels else train_labels
    print(f"[augment] combined train size: {len(combined_texts)} (original={len(train_texts)}, aug={len(aug_texts)})")

    output_dir = Path(cfg.paths.output_dir) / f"{model_key}_augmented_{'_'.join(aug_langs)}_all"
    model, tokenizer = build_model(model_key, num_labels=len(label_names))
    train_ds = build_hf_dataset(combined_texts, combined_labels, tokenizer, spec.max_length)
    dev_ds = build_hf_dataset(list(dev_texts), dev_labels, tokenizer, spec.max_length)

    arr = np.asarray(combined_labels, dtype="float32")
    pos_weight = ((len(arr) - arr.sum(0)) / np.maximum(arr.sum(0), 1)).tolist()
    print(f"[augment] pos_weight={[round(w, 2) for w in pos_weight]}")

    artifacts = train_transformer(
        model, tokenizer, train_ds, dev_ds,
        config=cfg, output_dir=output_dir,
        label_names=label_names, pos_weight=pos_weight,
    )

    (output_dir / "dev_metrics.json").write_text(json.dumps(artifacts.metrics, indent=2))
    (output_dir / "run_config.json").write_text(json.dumps({
        "model_key": model_key,
        "augmented_languages": aug_langs,
        "n_original": len(train_texts),
        "n_augmented": len(aug_texts),
        "n_combined": len(combined_texts),
    }, indent=2))
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    print(f"[augment] dev f1_macro={artifacts.metrics.get('f1_macro'):.4f}")
    print(f"[augment] artifacts -> {output_dir}")


if __name__ == "__main__":
    main()
