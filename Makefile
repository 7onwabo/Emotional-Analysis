.PHONY: help setup setup-shap install install-dev download train eval test lint format clean

# Interpreter used to CREATE the venv. Must be a native build with torch wheels:
#   Apple Silicon -> arm64 Python 3.12 (e.g. `brew install python@3.12`)
#   Windows / Linux -> Python 3.10-3.12
# Override on the CLI: `make setup PYTHON=python3.12`
PYTHON ?= python3.12
VENV_PY := .venv/bin/python

help:
	@echo "Targets:"
	@echo "  setup       create .venv (PYTHON=$(PYTHON)) + install dev,explain,augment deps"
	@echo "  setup-shap  add SHAP into the venv (may build llvmlite from source)"
	@echo "  install     install runtime deps (editable)"
	@echo "  install-dev install runtime + dev deps"
	@echo "  download    fetch BRIGHTER datasets into ./data"
	@echo "  train       run training (configs/training.yaml)"
	@echo "  eval        run evaluation on test split"
	@echo "  test        run pytest"
	@echo "  lint        ruff + mypy"
	@echo "  format      black + ruff --fix"
	@echo "  clean       remove caches / build artifacts"

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PY) -m pip install -U pip
	$(VENV_PY) -m pip install -e ".[dev,explain,augment]"
	$(VENV_PY) -c "import platform,torch,transformers; print('arch', platform.machine(), '| torch', torch.__version__, '| transformers', transformers.__version__)"

setup-shap:
	$(VENV_PY) -m pip install -e ".[shap]"

install:
	$(VENV_PY) -m pip install -e .

install-dev:
	$(VENV_PY) -m pip install -e ".[dev,explain,augment,track]"

download:
	$(VENV_PY) scripts/download_data.py

train:
	$(VENV_PY) scripts/train.py

eval:
	$(VENV_PY) scripts/evaluate.py

test:
	$(VENV_PY) -m pytest -q

lint:
	$(VENV_PY) -m ruff check src tests scripts
	$(VENV_PY) -m mypy src

format:
	$(VENV_PY) -m black src tests scripts
	$(VENV_PY) -m ruff check --fix src tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
