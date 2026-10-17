# LegalEval Extract: simple solution

One script that predicts the expert score (1-5) for each extracted block of a court decision.

## How it works

1. **Features** (one row per case and block): extraction length, how much of the extraction's words and
   three-word phrases appear in the decision, TF-IDF similarity to the best-matching 50-word window of the
   decision, where in the decision that window sits, and which block it is.
2. **Model**: 0.6 x gradient boosting (150 trees) + 0.4 x ridge regression, averaged.
3. **Validation**: 5-fold cross-validation grouped by case (all 7 blocks of a case stay in the same fold,
   like the real train/test split).
4. **Rescaling**: predictions are stretched to the mean and spread of the expert scores, because the
   metric's CCC part punishes predictions that are bunched around the average.

## Run

Put `train.csv`, `test.csv` and `sample_submission.csv` in `../data/`, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

The submission is written to `outputs/submission.csv`.

## Score

The script prints the cross-validation score using the competition formula
(0.55 x CCC + 0.30 x Pearson + 0.15 x normalized MAE; the MAE part is assumed to be `1 - MAE / 4`).
With only 101 training cases the number moves by about +/-0.01 between different fold splits;
averaged over 10 different splits it is about 0.758.
