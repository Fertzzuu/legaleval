# Judicial extraction quality competition starter

## Data
Place the original competition files inside `data/`:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Stage 1: exact previously used baseline
```bash
python baseline.py --mode basic
```
Outputs: `outputs/oof_basic.csv`, `outputs/submission_basic.csv`.

## Stage 1b: add source/extraction numeric quality features
```bash
python baseline.py --mode enhanced
```
Outputs: `outputs/oof_enhanced.csv`, `outputs/submission_enhanced.csv`.
Evaluate grouped cross-validation output `MAE`, `Pearson`, `CCC`.
Compare results. New features may not improve score.

## Stage 2: semantic encoder (downloads model on first run)
```bash
pip install -r requirements-embeddings.txt
python embeddings.py --blend-weight 0.5
```
Outputs: `outputs/oof_embeddings.csv`, `outputs/submission_embeddings.csv`
and (if enhanced baseline outputs exist) `outputs/submission_blended.csv`.
In subsequent runs, embeddings are reused from cache if inputs/model match.

## Stage 3: submission
Submit your best `outputs/submission_*.csv`. All have precisely
`row_id,predicted_score` in `sample_submission.csv` order.
Do not submit OOF files (they contain predictions for the training set).

## Validation rules
- `GroupKFold(n_splits=5)` groups by `id_sud_resh`.
- All TF-IDF, scaling and one-hot encoders fit inside each CV fold.
- All prediction scores are clipped to [1, 5].
- Semantic encoder is fixed/pretrained; no supervision on test labels.
- Choosing a model on 5-fold OOF estimates can be noisy on 101 cases.
- Do not tune many hyperparameters on the same folds without final re-check.

Requires model download (internet) for embedding stage. On CPU this stage
will be substantially slower than Stage 1; a GPU is recommended.
