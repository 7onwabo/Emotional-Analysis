# Data

Layout:

- `raw/` — direct downloads from HF Hub. Never edit. **Gitignored.**
- `processed/` — tokenised / split / cleaned. Regenerable from `raw/`. **Gitignored.**
- `external/` — extras (AfriSenti, AfriHate, Sesotho headlines) used for auxiliary tasks.

## Provenance

| Dataset | Source | License | Languages used |
|---|---|---|---|
| BRIGHTER | HuggingFace Hub | CC BY 4.0 | zul, xho, afr |
| EthioEmo | HuggingFace Hub | check repo | — (Ethiopian langs, used for transfer only) |
| AfriSenti | HuggingFace Hub | CC BY 4.0 | auxiliary |
| AfriHate | HuggingFace Hub | check repo | auxiliary |

Run `make download` to populate `raw/`.
