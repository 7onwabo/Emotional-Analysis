"""Train a model on the configured language(s).

Run: `python scripts/train.py model=afro_xlmr_base language=zul train.num_epochs=3`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_analysis.utils.config import load_config  # noqa: E402
from emotion_analysis.utils.seeds import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train emotion classifier")
    p.add_argument("--model", default=None, help="Model key from configs/models.yaml")
    p.add_argument("--language", default="zul", help="Target language ISO code")
    p.add_argument("--override", nargs="*", default=[], help="Dotlist overrides, e.g. train.batch_size=8")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("training", "data", "languages", "models", overrides=args.override)
    set_seed(cfg.seed)

    model_key = args.model or cfg.default
    print(f"[train] model={model_key} language={args.language} seed={cfg.seed}")

    raise NotImplementedError(
        "TODO: build dataset -> build model -> training.trainer.train_transformer(...)"
    )


if __name__ == "__main__":
    main()
