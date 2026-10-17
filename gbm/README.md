# Feature-GBM baseline

Hand-built text features + scikit-learn `HistGradientBoostingRegressor`, scored with repeated
`GroupKFold` by case on the competition metric. Every run compares four feature sets on identical folds:
`lexical`, `lexical+cross_block`, `lexical+embedding` and `all` (the submission uses `--submit-set`,
default `all`).

Embedding features use `intfloat/multilingual-e5-large` through `fastembed` (CPU, ONNX). The first run
embeds ~31k texts (~26 minutes on an M4 Max); vectors are cached by text hash, so later runs take seconds.

Expects the competition files in `../data/` (override with `--data-dir`).

```bash
uv sync --all-groups
uv run legaleval-gbm
uv run pytest
uv run jupyter lab notebooks/explore.ipynb
```

`legaleval-gbm` writes to `outputs/` (gitignored):

| File | Contents |
|------|----------|
| `features_train.csv`, `features_test.csv` | One row per (case, block) with every feature |
| `oof_gbm.csv` | Out-of-fold predictions of the submit model + set, averaged over seeds, raw and rescaled |
| `importance_gbm.csv` | Permutation importance on held-out folds (submit model + set) |
| `embeddings/multilingual-e5-large.npz` | Unit-normalized E5 vectors keyed by SHA-1 of the prefixed text |
| `submission_<model>_<set>_<timestamp>.csv` | Ready for `kaggle competitions submit -c legaleval-extract -f <file> -m "<message>"` |

## Layout

| Module | Role |
|--------|------|
| `data.py` | Loading, column names, the 7 block names |
| `metric.py` | Competition score (0.55 CCC + 0.30 Pearson + 0.15 normalized MAE) and rescaling |
| `text.py` | Lemmatization (pymorphy3), trigrams, sliding windows |
| `features.py` | Lexical, cross-block and embedding features; the `FEATURE_SETS` compared in CV |
| `embeddings.py` | E5 model + resumable on-disk embedding cache |
| `cv.py` | Block-prior baseline, GBM factory, repeated grouped CV, permutation importance |
| `cli.py` | The `legaleval-gbm` command |

The normalized MAE term assumes `1 - MAE / 4`; the host does not define it.
