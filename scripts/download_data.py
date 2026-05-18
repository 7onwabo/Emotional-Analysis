"""Download BRIGHTER + EthioEmo + auxiliary datasets into ./data/raw.

Run: `python scripts/download_data.py`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_analysis.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download datasets")
    p.add_argument("--languages", nargs="+", default=None, help="ISO codes to download (default: all targets)")
    p.add_argument("--include-auxiliary", action="store_true", help="Also fetch AfriSenti / AfriHate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("data", "languages")

    target_codes = args.languages or [lang.code for lang in cfg.target_languages]
    print(f"[download] target languages: {target_codes}")
    print(f"[download] raw dir: {cfg.paths.raw}")

    raise NotImplementedError(
        "TODO: for each lang call datasets.load_dataset(BRIGHTER_HF_ID, lang) and save_to_disk"
    )


if __name__ == "__main__":
    main()
