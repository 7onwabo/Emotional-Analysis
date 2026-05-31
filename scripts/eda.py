"""Exploratory data analysis for the BRIGHTER target-language splits.

Produces:
    reports/eda/brighter_summary.json   
    reports/eda/brighter_summary.md     

Run after `scripts/download_data.py`:
    python scripts/eda.py
    python scripts/eda.py --languages afr
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emotion_analysis import EMOTION_LABELS  
from emotion_analysis.data.loaders import load_brighter  
from emotion_analysis.utils.config import load_config  


def _bucket(n: int) -> str:
    if n <= 32:
        return "0-32"
    if n <= 64:
        return "33-64"
    if n <= 128:
        return "65-128"
    if n <= 256:
        return "129-256"
    if n <= 512:
        return "257-512"
    return "513+"


def analyse_split(ds: Any, label_cols: list[str], text_col: str = "text") -> dict[str, Any]:
    n = len(ds)
    label_counts = {c: {"present": 0, "absent": 0, "null": 0} for c in label_cols}
    cooccurrence: Counter[tuple[str, str]] = Counter()
    char_buckets: Counter[str] = Counter()
    tok_buckets: Counter[str] = Counter()
    empty = 0
    seen: dict[str, int] = {}
    duplicates = 0
    label_density: list[int] = []

    for row in ds:
        present_here: list[str] = []
        for c in label_cols:
            v = row.get(c) if c in ds.column_names else None
            if v is None:
                label_counts[c]["null"] += 1
            elif int(v) == 1:
                label_counts[c]["present"] += 1
                present_here.append(c)
            else:
                label_counts[c]["absent"] += 1
        label_density.append(len(present_here))
        for i, a in enumerate(present_here):
            for b in present_here[i:]:
                key = tuple(sorted((a, b)))
                cooccurrence[key] += 1

        text = (row.get(text_col) or "").strip()
        if not text:
            empty += 1
            continue
        char_buckets[_bucket(len(text))] += 1
        tok_buckets[_bucket(len(text.split()))] += 1
        seen[text] = seen.get(text, 0) + 1

    duplicates = sum(c - 1 for c in seen.values() if c > 1)

    return {
        "rows": n,
        "empty_text": empty,
        "duplicate_text_rows": duplicates,
        "label_counts": label_counts,
        "label_prevalence": {
            c: round(label_counts[c]["present"] / max(n, 1), 4) for c in label_cols
        },
        "mean_labels_per_example": round(sum(label_density) / max(n, 1), 3),
        "label_density_hist": dict(Counter(label_density)),
        "cooccurrence": {f"{a}|{b}": v for (a, b), v in sorted(cooccurrence.items())},
        "char_len_hist": dict(char_buckets),
        "token_len_hist": dict(tok_buckets),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# BRIGHTER EDA — per-language summary", ""]
    for lang, by_split in summary["languages"].items():
        lines.append(f"## {lang}")
        for split, stats in by_split.items():
            lines.append(f"### `{split}` — {stats['rows']} rows "
                         f"(empty={stats['empty_text']}, dupes={stats['duplicate_text_rows']})")
            lines.append("")
            lines.append("| label | present | absent | null | prevalence |")
            lines.append("|---|---:|---:|---:|---:|")
            for c, counts in stats["label_counts"].items():
                prev = stats["label_prevalence"][c]
                lines.append(f"| {c} | {counts['present']} | {counts['absent']} | {counts['null']} | {prev:.3f} |")
            lines.append("")
            lines.append(f"- mean labels/example: **{stats['mean_labels_per_example']}**")
            lines.append(f"- char-len buckets: {stats['char_len_hist']}")
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA over downloaded BRIGHTER splits")
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--out-dir", default="reports/eda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config("data", "languages")
    label_cols = list(cfg.datasets.brighter.label_columns) or EMOTION_LABELS
    target_codes = args.languages or [lang.brighter_config for lang in cfg.target_languages]
    splits = args.splits or list(cfg.datasets.brighter.splits)
    raw_dir = Path(cfg.paths.raw) / "brighter"

    summary: dict[str, Any] = {"label_columns": label_cols, "languages": {}}
    for lang in target_codes:
        summary["languages"][lang] = {}
        for split in splits:
            print(f"[eda] {lang}/{split}")
            ds = load_brighter(lang, split=split, raw_dir=raw_dir, hub_fallback=False)
            summary["languages"][lang][split] = analyse_split(ds, label_cols)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "brighter_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "brighter_summary.md").write_text(render_markdown(summary))
    print(f"[eda] wrote {out_dir}/brighter_summary.{{json,md}}")


if __name__ == "__main__":
    main()
