# Getting Started — Step by Step

Onboarding for new team members. Follow top to bottom. You'll go from a fresh clone to trained models and evaluation reports.

Project: multilabel emotion classification for 5 African languages — **afr, swa, hau** (BRIGHTER, Latin script) and **amh, tir** (EthioEmo, Ge'ez script). Labels: `anger, disgust, fear, joy, sadness, surprise`.

---

## 0. Prerequisites

- **git** and **make** installed.
- A **native Python 3.10–3.12 interpreter that has PyTorch wheels.** This is the #1 thing people get wrong — read the table:

| Platform | Use this interpreter | Do NOT use |
|---|---|---|
| **Apple Silicon (M1/M2/M3)** | native **arm64** Python 3.12 — install with `brew install python@3.12`, then `python3.12` | x86_64 / Rosetta Anaconda (torch caps at 2.2.2, builds fail); Python 3.13/3.14 (no torch wheels yet) |
| **Windows** | Python 3.10–3.12 from python.org (`py -3.12`) | 3.13+ |
| **Linux** | Python 3.10–3.12 | 3.13+ |

**Check your interpreter is correct before anything else:**

```bash
# macOS Apple Silicon — MUST print: arm64 3.12.x
python3.12 -c "import platform,sys; print(platform.machine(), sys.version.split()[0])"
```

```powershell
:: Windows — should print 3.10–3.12
py -3.12 -c "import platform,sys; print(platform.machine(), sys.version.split()[0])"
```

If Apple Silicon prints `x86_64`, you're in a Rosetta/Anaconda shell — open a normal Terminal and use the homebrew `python3.12`, do not use `conda`'s python.

---

## 1. Clone (already done if you're reading this)

```bash
git clone <repo-url>
cd Emotional-Analysis
git checkout dev          # active development branch
```

---

## 2. Create the environment and install

`make setup` creates a `.venv/` and installs the package + dev/explain/augment dependencies. It defaults to `python3.12`.

```bash
# macOS Apple Silicon (default)
make setup

# Windows / Linux, or to force a specific interpreter:
make setup PYTHON=py        # Windows (or PYTHON=python)
make setup PYTHON=python3.11
```

On success it prints an arch/version check, e.g.:

```
arch arm64 | torch 2.x.x | transformers 5.x.x
```

That line must show your real arch and a torch version, with **no** "Disabling PyTorch" warning.

> **SHAP is intentionally NOT installed** (it pulls `numba`/`llvmlite`, which build from source and often fail). LIME + Captum cover explainability. If you specifically need SHAP later: `make setup-shap`.

### If `make setup` fails

- **`torch`/`llvmlite` build errors** → wrong interpreter (x86_64 or 3.13+). Fix: `rm -rf .venv` then re-run with a correct `PYTHON=` (see the table in step 0).
- **"Disabling PyTorch ... requires 2.4"** → you're on an old torch (x86_64 mac). Use a native arm64 Python.

---

## 3. (Optional) HuggingFace token

The datasets are public — **you do not need a token to download them.** A token only raises rate limits.

```bash
cp .env.example .env       # optional; only set HF_TOKEN if you hit rate limits
```

---

## 4. Download the data

Pulls BRIGHTER (afr/swa/hau) + EthioEmo (amh/tir) into `data/raw/{brighter,ethioemo}/`.

```bash
make download
# or a subset:
.venv/bin/python scripts/download_data.py --languages afr amh
```

Writes a `manifest.json` per source (row counts + null-label audit). Re-run with `--force` to ignore the local cache.

> Windows note: there is no `.venv/bin/` — use `.venv\Scripts\python.exe` instead of `.venv/bin/python` in every command below (or `.venv\Scripts\activate` once, then just `python`).

---

## 5. Sanity check — run the tests

```bash
make test
```

Expect `20 passed`. If tests fail here, stop and fix the environment before training — something is wrong with the install.

---

## 6. Explore the data (optional but recommended)

```bash
.venv/bin/python scripts/eda.py
```

Writes per-language label frequencies, co-occurrence, and text-length stats to `reports/eda/brighter_summary.{json,md}`.

---

## 7. Train a model

Models are keyed in `configs/models.yaml`. `--language all` pools all 5 targets; a single code (e.g. `afr`) trains one language.

### 7a. Classical baseline (fast — start here to confirm the pipeline)

```bash
.venv/bin/python scripts/train.py --model tfidf_logreg --language afr
```

Trains in seconds. Writes `outputs/tfidf_logreg_afr/` (fitted `pipeline.joblib` + `dev_metrics.json`).

### 7b. Transformer fine-tune (AfroXLMR)

**On 16 GB Apple Silicon, the default config runs out of memory.** Use the low-RAM flags (gradient checkpointing + frozen base layers + small batch):

```bash
.venv/bin/python scripts/train.py --model afro_xlmr_base --language all \
  --override train.batch_size=4 train.eval_batch_size=8 train.gradient_accumulation_steps=4 \
             train.num_epochs=3 train.gradient_checkpointing=true train.freeze_base_layers=6
```

- Takes ~15–20 min on an M2 for all 5 languages.
- Writes `outputs/afro_xlmr_base_all/` (HF checkpoint + `dev_metrics.json` + `run_config.json`).
- **24 GB+ machine?** Drop `gradient_checkpointing`/`freeze_base_layers` and raise `batch_size` to 8–16 for faster training.

Other transformer keys you can swap into `--model`: `afriberta_large`, `xlm_roberta_base`, `mbert`.

---

## 8. Evaluate (per-language report)

```bash
.venv/bin/python scripts/evaluate.py --checkpoint outputs/afro_xlmr_base_all --language all
```

Prints macro/micro F1 per language and writes `reports/eval/{checkpoint}/test_report.json` (per-language metrics, per-label confusion) + `test_error_examples.json` (worst misclassifications). Auto-detects baseline vs transformer checkpoints.

---

## 9. Explain predictions (error analysis)

```bash
# LIME (works for baseline + transformer)
.venv/bin/python scripts/explain.py --checkpoint outputs/tfidf_logreg_afr --method lime --language afr

# Captum integrated gradients (transformer only)
.venv/bin/python scripts/explain.py --checkpoint outputs/afro_xlmr_base_all --method ig --language amh
```

Writes token attributions for the worst-misclassified examples to `reports/explain/{checkpoint}/`.

---

## Command cheat-sheet

| Goal | Command |
|---|---|
| Install | `make setup` (or `make setup PYTHON=...`) |
| Download data | `make download` |
| Run tests | `make test` |
| EDA | `.venv/bin/python scripts/eda.py` |
| Train baseline | `.venv/bin/python scripts/train.py --model tfidf_logreg --language afr` |
| Train transformer (16 GB) | see step 7b |
| Evaluate | `.venv/bin/python scripts/evaluate.py --checkpoint <dir> --language all` |
| Explain | `.venv/bin/python scripts/explain.py --checkpoint <dir> --method lime` |
| Lint / format | `make lint` / `make format` |

---

## Where things live

```
configs/        YAML config — change languages/labels/hyperparams HERE, not in code
  languages.yaml   the 5 targets (each has a `source: brighter|ethioemo`)
  models.yaml      model registry (baseline + transformer keys)
  training.yaml    hyperparameters + low-RAM flags
  data.yaml        dataset ids + preprocessing
data/raw/       downloaded datasets (gitignored)
outputs/        trained models + dev metrics (gitignored)
reports/        eda / eval / explain outputs
scripts/        download_data, eda, train, evaluate, explain
src/emotion_analysis/   the package (data, models, training, evaluation)
tests/          pytest suite
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `make setup` fails building torch/llvmlite | x86_64 or Python 3.13+ interpreter | `rm -rf .venv`; re-run with native arm64 3.12 / `PYTHON=` |
| "Disabling PyTorch ... 2.4 required" | old torch on x86_64 mac | use native arm64 Python |
| `MPS backend out of memory` | 16 GB Mac, batch too big | use the step 7b low-RAM flags; close other apps |
| `Trainer got unexpected kwarg 'tokenizer'` | shouldn't happen (handled), but if editing trainer: transformers 5.x renamed it to `processing_class` | n/a — already version-agnostic |
| `.venv/bin/python: not found` on Windows | wrong path | use `.venv\Scripts\python.exe` |
| Tests fail right after install | broken environment | re-check interpreter (step 0), `rm -rf .venv`, reinstall |

Stuck? Post in the team channel with the exact command and the full error.
