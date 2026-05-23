# Emotion Analysis — African Languages

Multilabel emotion classification across five African languages from **BRIGHTER** (Afrikaans, Swahili, Hausa) and **EthioEmo** (Amharic, Tigrinya), spanning **Latin and Ge'ez scripts**.

COS 760 group project — University of Pretoria.

## Target languages

Five languages across two datasets, two scripts, and three families — chosen for direct-fine-tune-able train splits and maximal contrast for the RQ2 linguistic-factor analysis.

| Language | Code | Source | Script | Train | Dev | Test | Family | Region |
|---|---|---|---|---|---|---|---|---|
| Afrikaans | `afr` | BRIGHTER | Latin | 1,222 | 196 | 2,130 | Indo-European (Germanic) | South Africa |
| Swahili | `swa` | BRIGHTER | Latin | 3,307 | 1,102 | 3,312 | Niger-Congo (Bantu) | East Africa |
| Hausa | `hau` | BRIGHTER | Latin | 2,145 | 712 | 2,160 | Afro-Asiatic (Chadic) | West Africa |
| Amharic | `amh` | EthioEmo | Ge'ez | 3,549 | 592 | 1,774 | Afro-Asiatic (Semitic) | Ethiopia |
| Tigrinya | `tir` | EthioEmo | Ge'ez | 3,681 | 614 | 1,840 | Afro-Asiatic (Semitic) | Ethiopia / Eritrea |

Both datasets share the **same 6-label int64 multilabel schema** (`id, text, anger…surprise`), so one loader serves both. The Latin-vs-Ge'ez script split and the resource/morphology spread are central to RQ2 (what linguistic factors explain performance gaps). isiZulu/isiXhosa were dropped earlier (BRIGHTER `train=0` — no direct fine-tuning); EthioEmo's `orm`/`som` (both Latin) remain available if the set is widened.

## Emotion labels (BRIGHTER multilabel schema)

`anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise` — multi-hot per text. Intensity prediction (0–3) is a stretch goal if a target language has the intensity split.

## Project layout

```
.
├── configs/                    # Hydra / YAML configs
│   ├── data.yaml
│   ├── languages.yaml
│   ├── models.yaml
│   └── training.yaml
├── data/
│   ├── raw/                    # raw downloads (gitignored)
│   ├── processed/              # tokenised / split (gitignored)
│   └── external/               # AfriSenti, AfriHate, etc.
├── notebooks/                  # EDA, error analysis
├── reports/                    # paper draft, figures
├── scripts/
│   ├── download_data.py        # ✅ pulls BRIGHTER -> data/raw/brighter
│   ├── eda.py                  # ✅ per-lang stats -> reports/eda
│   ├── train.py                # ✅ baseline + transformer training
│   ├── evaluate.py             # ✅ per-language test report + error tables
│   └── explain.py              # ✅ LIME / Captum IG token attributions
├── src/emotion_analysis/
│   ├── data/                   # loaders, preprocessing, augmentation
│   ├── models/                 # baselines + transformer wrappers
│   ├── training/               # trainer, metrics
│   ├── evaluation/             # classification report, SHAP/LIME
│   └── utils/                  # config, seed, logging
└── tests/
```

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | HF ecosystem |
| DL | PyTorch + HuggingFace `transformers` | standard for fine-tuning |
| Data | HuggingFace `datasets`, `pandas` | BRIGHTER ships on HF Hub |
| Configs | Hydra + OmegaConf | reproducible sweeps |
| Baselines | scikit-learn (LogReg/SVM on TF-IDF) | cheap reference point |
| Transformers | AfroXLMR-base, AfriBERTa, XLM-R, mBERT | covers African-tuned + multilingual generic |
| Multilabel split | `iterative-stratification` | preserves label distribution |
| Augmentation | back-translation (`deep-translator`), `nlpaug` | low-resource boost |
| Explainability | SHAP, LIME, Captum (attention) | error analysis required by rubric |
| Tracking | Weights & Biases (optional) | experiment logs |
| Tests | pytest, ruff, mypy | hygiene |

## Project status — for teammates

We work in **phases**. Phase 1 (data layer) is **done**; phases 2+ (modeling, eval, report) are next.

| Phase | What | Status |
|---|---|---|
| **1. Data** | download BRIGHTER, load → clean multi-hot `EmotionExample`, EDA | ✅ done |
| **2. Models** | sklearn baselines + transformer fine-tune + trainer + train/eval scripts | ✅ done (baseline + transformer trained) |
| **3. Eval+** | per-language F1 + confusion + error tables ✅; LIME + Captum IG explainability ✅; SHAP optional | ✅ done |
| 4. Report | ACL-template paper, responsible-NLP reflection | ⏳ |

> **RQ coverage:** RQ1/RQ2 are served by the BRIGHTER + transformer + per-language report path above. Two datasets named in the RQs are **not yet wired**: **EthioEmo** (RQ1) and **AfriSenti/AfriHate** (RQ3, transfer/augmentation). These are required, not optional — see the roadmap.

### What phase 1 gives you

- `scripts/download_data.py` — pulls BRIGHTER `{afr,swa,hau}` × `{train,dev,test}` from HF Hub, saves to `data/raw/brighter/{lang}/{split}` via `datasets.save_to_disk`, and writes `data/raw/brighter/manifest.json` (row counts + **null-label audit** — BRIGHTER leaves some emotions, e.g. afr `surprise`, as `null`).
- `src/emotion_analysis/data/loaders.py` — `load_brighter` (reads local copy first, falls back to hub), `to_emotion_examples` (null/missing label → `0`, fixed `EMOTION_LABELS` order), `to_arrays`, `load_brighter_examples`.
- `src/emotion_analysis/data/preprocessing.py` — `preprocess_text` (strip URLs/mentions, whitespace) + `preprocess_examples` (drops `<min_chars`, clips `>max_chars`).
- `scripts/eda.py` — per-(lang,split) label freq/prevalence/null, label co-occurrence, text-length histograms → `reports/eda/brighter_summary.{json,md}`.
- `tests/test_data.py` — offline tests (in-memory HF dataset, no network).

All knobs live in `configs/*.yaml` — change languages/labels/preprocessing there, not in code.

## Setup — read before installing

You need a **native Python 3.10–3.12 with PyTorch wheels**. Picking the wrong interpreter is the #1 setup failure:

| Platform | Use | Avoid |
|---|---|---|
| **Apple Silicon (M1/M2/M3)** | native **arm64** Python 3.12 — `brew install python@3.12` → `python3.12` | x86_64 / Rosetta Anaconda (`platform.machine()=='x86_64'`) → torch caps at 2.2.2, llvmlite won't build; Python 3.13/3.14 → no torch wheels yet |
| **Windows / Linux** | Python 3.10–3.12 (`python` / `py -3.12`) | 3.13+ until torch ships wheels |

Check your interpreter first:

```bash
python3.12 -c "import platform,sys; print(platform.machine(), sys.version.split()[0])"
# Apple Silicon must print: arm64 3.12.x
```

`make setup` defaults to `python3.12`. Override per machine: `make setup PYTHON=python` (Windows) or `make setup PYTHON=python3.11`.

## Quick start

```bash
make setup                          # .venv via $(PYTHON) + install .[dev,explain,augment]; prints arch/torch check
cp .env.example .env                # add HF_TOKEN only if a dataset is gated

make download                       # pull BRIGHTER -> data/raw/brighter
.venv/bin/python scripts/eda.py     # stats -> reports/eda
make test                           # run pytest in the venv

# phase 2 — train + evaluate:
.venv/bin/python scripts/train.py --model tfidf_logreg --language afr     # cheap baseline (start here)
.venv/bin/python scripts/train.py --model afro_xlmr_base --language all    # transformer
.venv/bin/python scripts/evaluate.py --checkpoint outputs/tfidf_logreg_afr --language afr
.venv/bin/python scripts/explain.py --checkpoint outputs/tfidf_logreg_afr --method lime   # token attributions
```

**Low-RAM training (16 GB Apple Silicon).** The default transformer config OOMs on 16 GB MPS. Use gradient checkpointing + a small batch:

```bash
.venv/bin/python scripts/train.py --model afro_xlmr_base --language all \
  --override train.batch_size=4 train.eval_batch_size=8 train.gradient_accumulation_steps=4 \
             train.num_epochs=3 train.gradient_checkpointing=true train.freeze_base_layers=6
```

`gradient_checkpointing` recomputes activations (big memory cut, ~20% slower); `freeze_base_layers=N` freezes embeddings + the bottom N encoder layers (less optimizer/grad memory). 24 GB+ machines can drop these and raise `batch_size`.

> If `make setup` fails on torch/llvmlite, you almost certainly used an x86_64 or 3.13+ interpreter — `rm -rf .venv` and re-run with a correct `PYTHON=` (see table above).

Training writes to `outputs/{model}_{lang}/` (fitted pipeline or HF checkpoint +
`dev_metrics.json` + `run_config.json`). Evaluation writes per-language reports
and error tables to `reports/eval/{checkpoint}/`. `--language all` pools afr+swa+hau;
a single ISO code trains/evaluates one language.

Useful flags:

```bash
# one language only
.venv/bin/python scripts/download_data.py --languages afr
# re-pull (ignore local cache)
.venv/bin/python scripts/download_data.py --force
```

**Optional extras:**

- `make setup-shap` — adds SHAP (phase 3 explainability). Kept out of the default install because it pulls `numba`/`llvmlite`, which build from source on some toolchains. LIME + Captum (in `[explain]`) are wheel-only and cover most of the analysis.
- EthioEmo / AfriSenti / AfriHate loaders are deferred; BRIGHTER alone covers all three target languages.

## Models on roadmap

- **Baseline**: TF-IDF + Logistic Regression (one-vs-rest), TF-IDF + Linear SVM.
- **Transformer fine-tune**: `Davlan/afro-xlmr-base`, `castorini/afriberta_large`, `xlm-roberta-base`, `bert-base-multilingual-cased`.
- **Stretch**: adapter-based (`adapters` library), LLM zero/few-shot annotator comparison.

## Datasets

| Dataset | Use | Source |
|---|---|---|
| BRIGHTER | primary multilabel emotion | HF Hub (`brighter-dataset`) |
| EthioEmo | extra African coverage | HF Hub |
| AfriSenti | sentiment auxiliary task / pretraining | HF Hub (`shmuhammad/AfriSenti`) |
| AfriHate | adjacent classification | HF Hub |
| Sesotho News Headlines | stretch — out-of-BRIGHTER SA lang | GitHub |

## Evaluation

- Macro / Micro F1, per-label F1, Precision, Recall (HF `evaluate`).
- Per-language breakdown (equity audit — required by rubric).
- SHAP + LIME on misclassified examples; attention rollout via Captum.

## Responsible NLP

See `reports/responsible-nlp.md` (to be written) — language coverage, data licensing (BRIGHTER is CC BY 4.0), annotator demographics, error fairness across the three target languages.

## Timeline

- **27 May 2026** — Report + presentation due. Project window: Apr–May.

## License

See `LICENSE`.
