"""Cross-lingual transfer evaluation (RQ2).
Zero-shot: train on a source language set, evaluate on withheld target languages.
Few-shot:  fine-tune the zero-shot model on N examples from the target language.
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
from emotion_analysis.data.datasets import build_split, resolve_languages
from emotion_analysis.evaluation.classification import per_language_report
from emotion_analysis.models.transformer import build_hf_dataset
from emotion_analysis.training.trainer import train_transformer
from emotion_analysis.utils.config import load_config
from emotion_analysis.utils.seeds import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-lingual transfer evaluation")
    p.add_argument("--train-langs", nargs="+", required=True, help="Languages to train on")
    p.add_argument("--eval-langs", nargs="+", required=True, help="Languages to evaluate zero/few-shot")
    p.add_argument("--model", default="afro_xlmr_base")
    p.add_argument("--few-shot", type=int, default=0, help="N examples per target lang for few-shot (0=zero-shot only)")
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models", overrides=args.override)
    set_seed(cfg.seed)

    model_key = args.model
    spec = cfg.models[model_key]
    label_names = list(cfg.task.labels) or EMOTION_LABELS
    all_targets = [lang.code for lang in cfg.target_languages]

    for lang in args.train_langs + args.eval_langs:
        if lang not in all_targets:
            raise SystemExit(f"Unknown language '{lang}'. Available: {all_targets}")

    tag = "zs" if args.few_shot == 0 else f"fs{args.few_shot}"
    src_tag = "+".join(args.train_langs)
    tgt_tag = "+".join(args.eval_langs)
    output_dir = Path(cfg.paths.output_dir) / f"{model_key}_xling_{src_tag}_to_{tgt_tag}_{tag}"

    print(f"[xling] train_langs={args.train_langs} eval_langs={args.eval_langs} few_shot={args.few_shot}")

    train_texts, train_labels, _ = build_split(cfg, args.train_langs, "train")
    dev_texts, dev_labels, _ = build_split(cfg, args.train_langs, "dev")
    print(f"[xling] source train={len(train_texts)} dev={len(dev_texts)}")

    from emotion_analysis.models.registry import build_model
    model, tokenizer = build_model(model_key, num_labels=len(label_names))

    train_ds = build_hf_dataset(train_texts, train_labels, tokenizer, spec.max_length)
    dev_ds = build_hf_dataset(dev_texts, dev_labels, tokenizer, spec.max_length)

    arr = np.asarray(train_labels, dtype="float32")
    pos_weight = ((len(arr) - arr.sum(0)) / np.maximum(arr.sum(0), 1)).tolist()

    artifacts = train_transformer(
        model, tokenizer, train_ds, dev_ds,
        config=cfg, output_dir=output_dir,
        label_names=label_names, pos_weight=pos_weight,
    )
    print(f"[xling] source dev f1_macro={artifacts.metrics.get('f1_macro'):.4f}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer as HFTokenizer

    def predict(ckpt: Path, texts: list[str], threshold: float) -> np.ndarray:
        tok = HFTokenizer.from_pretrained(str(ckpt))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        mdl.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i+32], truncation=True, max_length=128, padding=True, return_tensors="pt")
                logits = mdl(**enc).logits
                preds.append((torch.sigmoid(logits).numpy() >= threshold).astype(int))
        return np.vstack(preds)

    threshold = float(cfg.task.threshold)
    results: dict[str, dict] = {}

    print(f"\n[xling] === Zero-shot evaluation on {args.eval_langs} ===")
    test_texts, test_labels, test_langs = build_split(cfg, args.eval_langs, "test")
    preds = predict(output_dir, list(test_texts), threshold)
    zs_report = per_language_report(preds, test_labels, list(test_langs), label_names)
    results["zero_shot"] = zs_report

    print(f"[xling] zero-shot overall f1_macro={zs_report['overall']['f1_macro']:.4f}")
    for lang in args.eval_langs:
        if lang in zs_report:
            print(f"[xling]   {lang}: f1_macro={zs_report[lang]['f1_macro']:.4f} (n={zs_report[lang]['n_examples']})")

    if args.few_shot > 0:
        print(f"\n[xling] === Few-shot ({args.few_shot} examples/lang) fine-tune ===")
        rng = np.random.default_rng(cfg.seed)
        fs_texts_all, fs_labels_all = [], []
        for lang in args.eval_langs:
            lang_texts, lang_labels, _ = build_split(cfg, [lang], "train")
            n = min(args.few_shot, len(lang_texts))
            idx = rng.choice(len(lang_texts), size=n, replace=False)
            fs_texts_all += [lang_texts[i] for i in idx]
            fs_labels_all += [lang_labels[i] for i in idx]
            print(f"[xling]   few-shot {lang}: {n} examples")

        fs_texts_arr = np.array(fs_texts_all)
        fs_labels_arr = np.array(fs_labels_all)

        from emotion_analysis.models.registry import build_model as _build
        fs_model, fs_tokenizer = _build(model_key, num_labels=len(label_names))
        state = AutoModelForSequenceClassification.from_pretrained(str(output_dir)).state_dict()
        fs_model.load_state_dict(state)

        fs_train_ds = build_hf_dataset(fs_texts_all, fs_labels_arr, fs_tokenizer, spec.max_length)
        fs_dev_ds = build_hf_dataset(list(dev_texts), dev_labels, fs_tokenizer, spec.max_length)

        fs_output_dir = output_dir.parent / (output_dir.name.replace(f"_{tag}", f"_fs{args.few_shot}"))
        arr_fs = np.asarray(fs_labels_arr, dtype="float32")
        pw = ((len(arr_fs) - arr_fs.sum(0)) / np.maximum(arr_fs.sum(0), 1)).tolist()

        fs_artifacts = train_transformer(
            fs_model, fs_tokenizer, fs_train_ds, fs_dev_ds,
            config=cfg, output_dir=fs_output_dir,
            label_names=label_names, pos_weight=pw,
        )

        preds_fs = predict(fs_output_dir, list(test_texts), threshold)
        fs_report = per_language_report(preds_fs, test_labels, list(test_langs), label_names)
        results["few_shot"] = fs_report

        print(f"[xling] few-shot overall f1_macro={fs_report['overall']['f1_macro']:.4f}")
        for lang in args.eval_langs:
            if lang in fs_report:
                print(f"[xling]   {lang}: f1_macro={fs_report[lang]['f1_macro']:.4f}")

    out_dir = Path("reports/eval") / output_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_lingual_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[xling] results -> {out_dir}")


if __name__ == "__main__":
    main()
