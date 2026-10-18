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

## Run with Docker (Red Hat UBI)

The image is built from the bare Red Hat base image `registry.access.redhat.com/ubi9/ubi-minimal:9.8`.
Python 3.12 is installed from Red Hat's UBI repositories during the build; there is no other base or
build image. The competition data is not baked into the image: it is mounted at run time.

One command shows the empty base image, builds the solution from it (no cache) and runs it:

```bash
simple/demo.sh
```

Or step by step, from the repository root:

```bash
docker build -t legaleval-simple simple
docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/simple/outputs:/app/outputs" legaleval-simple
```

The container runs as user 1001. On a Linux host where that user cannot write to `simple/outputs`,
add `--user "$(id -u)"` to `docker run`.

## Score

The script prints the cross-validation score using the competition formula
(0.55 x CCC + 0.30 x Pearson + 0.15 x normalized MAE; the MAE part is assumed to be `1 - MAE / 4`).
With only 101 training cases the number moves by about +/-0.01 between different fold splits;
averaged over 10 different splits it is about 0.758.
