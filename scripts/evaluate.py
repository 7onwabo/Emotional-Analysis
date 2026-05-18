"""Evaluate a trained checkpoint on the test split.

Run: `python scripts/evaluate.py --checkpoint outputs/afro_xlmr_base_zul`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_analysis.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate emotion classifier")
    p.add_argument("--checkpoint", required=False, help="Path to trained model directory")
    p.add_argument("--language", default="zul", help="Language to evaluate on")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--explain", action="store_true", help="Run SHAP/LIME error analysis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages")
    print(f"[eval] ckpt={args.checkpoint} language={args.language} split={args.split}")

    raise NotImplementedError(
        "TODO: load checkpoint -> predict on split -> per_language_report -> save to reports/"
    )


if __name__ == "__main__":
    main()
