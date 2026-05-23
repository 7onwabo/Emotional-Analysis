"""Entry-point thin wrappers for the console scripts in pyproject.toml."""

from __future__ import annotations


def train() -> None:
    from scripts.train import main

    main()


def evaluate() -> None:
    from scripts.evaluate import main

    main()


def download() -> None:
    from scripts.download_data import main

    main()
