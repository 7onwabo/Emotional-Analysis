"""Download BRIGHTER emotion-categories splits for the project's target languages.

Saves each `{lang}/{split}` to `data/raw/brighter/` via `datasets.save_to_disk`
and writes `data/raw/brighter/manifest.json` summarising row counts and any
labels that are `null` in the raw data (BRIGHTER leaves un-annotated emotions
as null for some languages, e.g. `surprise` for afr).

Run:
    python scripts/download_data.py
    python scripts/download_data.py --languages afr swa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_analysis import EMOTION_LABELS  # noqa: E402
from emotion_analysis.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download BRIGHTER datasets")
    p.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="BRIGHTER language configs to download (default: configs/languages.yaml targets).",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to download (default: from configs/data.yaml).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local cache already populated.",
    )
    return p.parse_args()


def _null_label_audit(ds: Any, label_cols: list[str]) -> dict[str, int]:
    audit: dict[str, int] = {}
    for col in label_cols:
        if col not in ds.column_names:
            audit[col] = -1  # column absent for this language
            continue
        audit[col] = sum(1 for v in ds[col] if v is None)
    return audit


def download_language(
    hf_id: str,
    lang: str,
    splits: list[str],
    out_dir: Path,
    label_cols: list[str],
    *,
    force: bool,
) -> dict[str, Any]:
    from datasets import load_dataset, load_from_disk

    lang_dir = out_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"hf_id": hf_id, "lang": lang, "splits": {}}
    for split in splits:
        split_dir = lang_dir / split
        if split_dir.exists() and not force:
            print(f"  [skip] {lang}/{split} cached at {split_dir}")
            ds = load_from_disk(str(split_dir))
        else:
            print(f"  [pull] {lang}/{split} from {hf_id}")
            ds = load_dataset(hf_id, lang, split=split)
            ds.save_to_disk(str(split_dir))

        summary["splits"][split] = {
            "rows": len(ds),
            "columns": list(ds.column_names),
            "null_labels": _null_label_audit(ds, label_cols),
        }
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config("data", "languages")

    target_codes = args.languages or [lang.brighter_config for lang in cfg.target_languages]
    splits = args.splits or list(cfg.datasets.brighter.splits)
    label_cols = list(cfg.datasets.brighter.label_columns) or EMOTION_LABELS

    out_dir = Path(cfg.paths.raw) / "brighter"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] hf_id={cfg.datasets.brighter.hf_id}")
    print(f"[download] languages={target_codes} splits={splits}")
    print(f"[download] out_dir={out_dir}")

    manifest: dict[str, Any] = {
        "hf_id": cfg.datasets.brighter.hf_id,
        "label_columns": label_cols,
        "languages": {},
    }
    for lang in target_codes:
        manifest["languages"][lang] = download_language(
            hf_id=cfg.datasets.brighter.hf_id,
            lang=lang,
            splits=splits,
            out_dir=out_dir,
            label_cols=label_cols,
            force=args.force,
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[download] wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
