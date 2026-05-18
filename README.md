# Emotion Analysis — African Languages

Multilabel emotion classification for **Afrikaans (afr)**, **Swahili (swa)**, and **Hausa (hau)** using the BRIGHTER + EthioEmo dataset.

COS 760 group project — University of Pretoria.

## Why these three languages

BRIGHTER includes 17 African languages but only 11 carry a non-empty train split. isiZulu and isiXhosa were the obvious South African picks but both have `train=0` — direct fine-tuning is impossible without cross-lingual donor data. To keep the project scope realistic the targets were switched to languages with enough train data to fine-tune directly:

| Language | Config | Train | Dev | Test | Family | Region |
|---|---|---|---|---|---|---|
| Afrikaans | `afr` | 1,222 | 196 | 2,130 | Indo-European (Germanic) | South Africa |
| Swahili | `swa` | 3,307 | 1,102 | 3,312 | Niger-Congo (Bantu) | East Africa |
| Hausa | `hau` | 2,145 | 712 | 2,160 | Afro-Asiatic (Chadic) | West Africa |

This set keeps one South African language (Afrikaans), covers three distinct families, and spans three regions — enough contrast for the rubric's equity / fairness analysis without requiring cross-lingual transfer as a methodological dependency.

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
│   ├── download_data.py
│   ├── train.py
│   └── evaluate.py
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

## Quick start

```bash
make setup           # create .venv + install dev deps
cp .env.example .env # add HF_TOKEN if datasets gated
make download        # pull BRIGHTER + EthioEmo
make train           # train default config (AfroXLMR on zul)
make eval            # evaluate test split
```

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
