# Feature-GBM baseline

Hand-built text features + scikit-learn `HistGradientBoostingRegressor`, scored with repeated
`GroupKFold` by case on the competition metric.

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
| `oof_gbm.csv` | Out-of-fold predictions averaged over seeds, raw and rescaled |
| `importance_gbm.csv` | Permutation importance on held-out folds |
| `submission_gbm_<timestamp>.csv` | Ready for `kaggle competitions submit -c legaleval-extract -f <file> -m "<message>"` |

## Layout

| Module | Role |
|--------|------|
| `data.py` | Loading, column names, the 7 block names |
| `metric.py` | Competition score (0.55 CCC + 0.30 Pearson + 0.15 normalized MAE) and rescaling |
| `text.py` | Lemmatization (pymorphy3), trigrams, sliding windows |
| `features.py` | Per-row features against a per-case source index (TF-IDF, BM25) |
| `cv.py` | Block-prior baseline, GBM factory, repeated grouped CV, permutation importance |
| `cli.py` | The `legaleval-gbm` command |

The normalized MAE term assumes `1 - MAE / 4`; the host does not define it.
