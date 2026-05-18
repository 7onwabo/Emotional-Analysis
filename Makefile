.PHONY: help setup install install-dev download train eval test lint format clean

PYTHON := python
PIP := pip

help:
	@echo "Targets:"
	@echo "  setup       create venv and install dev deps"
	@echo "  install     install runtime deps (editable)"
	@echo "  install-dev install runtime + dev deps"
	@echo "  download    fetch BRIGHTER/EthioEmo datasets into ./data"
	@echo "  train       run training (configs/training.yaml)"
	@echo "  eval        run evaluation on test split"
	@echo "  test        run pytest"
	@echo "  lint        ruff + mypy"
	@echo "  format      black + ruff --fix"
	@echo "  clean       remove caches / build artifacts"

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(PIP) install -U pip && $(PIP) install -e ".[dev,explain,augment]"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,explain,augment,track]"

download:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) scripts/evaluate.py

test:
	pytest -q

lint:
	ruff check src tests scripts
	mypy src

format:
	black src tests scripts
	ruff check --fix src tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
