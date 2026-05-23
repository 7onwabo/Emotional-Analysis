"""Evaluate a trained checkpoint on a split, with per-language breakdown.

Examples:
    python scripts/evaluate.py --checkpoint outputs/tfidf_logreg_afr --language afr
    python scripts/evaluate.py --checkpoint outputs/afro_xlmr_base_all --language all

Writes reports/eval/{checkpoint_name}/{split}_report.json and error_examples.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_analysis import EMOTION_LABELS  # noqa: E402
from emotion_analysis.data.datasets import build_split, resolve_languages  # noqa: E402
from emotion_analysis.evaluation.classification import (  # noqa: E402
    collect_error_examples,
    confusion_per_label,
    per_language_report,
)
from emotion_analysis.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate emotion classifier")
    p.add_argument("--checkpoint", required=True, help="Path to trained model directory")
    p.add_argument("--language", default="all", help="ISO code or 'all'")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    p.add_argument("--explain", action="store_true", help="Run SHAP/LIME error analysis (phase 3)")
    return p.parse_args()


def predict_baseline(ckpt: Path, texts: list[str]) -> np.ndarray:
    import joblib

    pipeline = joblib.load(ckpt / "pipeline.joblib")
    return np.asarray(pipeline.predict(texts)).astype(int)


def predict_transformer(
    ckpt: Path, texts: list[str], max_length: int, threshold: float
) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(
                texts[i : i + batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs >= threshold).astype(int))
    return np.vstack(preds)


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models")
    ckpt = Path(args.checkpoint)
    label_names = list(cfg.task.labels) or EMOTION_LABELS
    threshold = args.threshold if args.threshold is not None else float(cfg.task.threshold)
    languages = resolve_languages(cfg, args.language)

    print(f"[eval] ckpt={ckpt} languages={languages} split={args.split} threshold={threshold}")
    texts, labels, langs = build_split(cfg, languages, args.split)
    print(f"[eval] {len(texts)} examples")

    is_baseline = (ckpt / "pipeline.joblib").exists()
    if is_baseline:
        preds = predict_baseline(ckpt, texts)
    else:
        preds = predict_transformer(ckpt, texts, max_length=128, threshold=threshold)

    report = per_language_report(preds, labels, list(langs), label_names)
    confusion = confusion_per_label(preds, labels, label_names)
    errors = collect_error_examples(texts, preds, labels, label_names, languages=list(langs), n=50)

    out_dir = Path("reports/eval") / ckpt.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.split}_report.json").write_text(
        json.dumps({"per_language": report, "confusion_per_label": confusion}, indent=2)
    )
    (out_dir / f"{args.split}_error_examples.json").write_text(json.dumps(errors, indent=2))

    print(f"[eval] overall f1_macro={report['overall']['f1_macro']:.4f} f1_micro={report['overall']['f1_micro']:.4f}")
    for lang in languages:
        if lang in report:
            zs = report[lang]["zero_support_labels"]
            zs_note = f" zero-support={zs}" if zs else ""
            print(
                f"[eval]   {lang}: f1_macro={report[lang]['f1_macro']:.4f} "
                f"f1_micro={report[lang]['f1_micro']:.4f} (n={report[lang]['n_examples']}){zs_note}"
            )
    print(f"[eval] report -> {out_dir}")

    if args.explain:
        print("[eval] --explain is phase 3 (SHAP/LIME) — skipped.")


if __name__ == "__main__":
    main()
