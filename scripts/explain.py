"""Token-level explanations for a trained checkpoint's predictions.

Picks the worst-misclassified examples on a split and attributes tokens for the
emotions in play (gold ∪ predicted), using LIME (model-agnostic) or, for
transformers, Captum integrated gradients.

Examples:
    python scripts/explain.py --checkpoint outputs/tfidf_logreg_afr --language afr
    python scripts/explain.py --checkpoint outputs/afro_xlmr_base_all --method ig --n 5

Writes reports/explain/{checkpoint}/{split}_{method}.json
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
from emotion_analysis.evaluation.explainability import (  # noqa: E402
    integrated_gradients,
    lime_explain,
    make_baseline_proba_fn,
    make_transformer_proba_fn,
)
from emotion_analysis.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain model predictions")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--language", default="all")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--method", default="lime", choices=["lime", "ig"])
    p.add_argument("--n", type=int, default=10, help="Number of examples to explain")
    p.add_argument("--num-features", type=int, default=10)
    p.add_argument("--max-length", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models")
    ckpt = Path(args.checkpoint)
    label_names = list(cfg.task.labels) or EMOTION_LABELS
    threshold = float(cfg.task.threshold)
    languages = resolve_languages(cfg, args.language)

    texts, labels, langs = build_split(cfg, languages, args.split)
    is_baseline = (ckpt / "pipeline.joblib").exists()

    if args.method == "ig" and is_baseline:
        raise SystemExit("Integrated gradients requires a transformer checkpoint, not a baseline.")

    if is_baseline:
        import joblib

        pipeline = joblib.load(ckpt / "pipeline.joblib")
        proba_fn = make_baseline_proba_fn(pipeline)
        model = tokenizer = None
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
        proba_fn = make_transformer_proba_fn(model, tokenizer, max_length=args.max_length)

    # Rank examples by number of wrong labels, explain the worst `n`.
    probs = proba_fn(texts)
    preds = (probs >= threshold).astype(int)
    n_wrong = (preds != labels).sum(axis=1)
    order = [i for i in np.argsort(-n_wrong) if n_wrong[i] > 0][: args.n]

    results = []
    for idx in order:
        gold = [label_names[j] for j in range(len(label_names)) if labels[idx, j] == 1]
        pred = [label_names[j] for j in range(len(label_names)) if preds[idx, j] == 1]
        focus = sorted(set(gold) | set(pred), key=label_names.index)
        attributions: dict[str, list] = {}
        for lbl in focus:
            li = label_names.index(lbl)
            if args.method == "lime":
                attributions[lbl] = lime_explain(
                    proba_fn, texts[idx], li, label_names, num_features=args.num_features
                )
            else:
                attributions[lbl] = integrated_gradients(
                    model, tokenizer, texts[idx], li, max_length=args.max_length
                )
        results.append(
            {
                "language": langs[idx],
                "text": texts[idx],
                "gold": gold,
                "pred": pred,
                "attributions": attributions,
            }
        )
        print(f"[explain] {langs[idx]} gold={gold} pred={pred}")

    out_dir = Path("reports/explain") / ckpt.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_{args.method}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[explain] {len(results)} explanations -> {out_path}")


if __name__ == "__main__":
    main()
