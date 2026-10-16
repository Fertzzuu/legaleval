import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from legaleval_gbm.cv import CvResult, block_prior, cv_permutation_importance, fit_predict, gbm, run_cv
from legaleval_gbm.data import CASE_ID, DEFAULT_DATA_DIR, ROW_ID, TARGET, load_sample_submission, load_split
from legaleval_gbm.features import FEATURE_COLUMNS, build_features
from legaleval_gbm.text import Lemmatizer

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-GBM baseline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--importance-repeats", type=int, default=5)
    return parser.parse_args()


def _summary(result: CvResult) -> list[dict[str, object]]:
    rows = []
    for variant, scores in (("raw", result.raw), ("rescaled", result.rescaled)):
        totals = np.array([s.total for s in scores])
        rows.append(
            {
                "model": result.name,
                "variant": variant,
                "total": totals.mean(),
                "total_std": totals.std(),
                "ccc": np.mean([s.ccc for s in scores]),
                "pearson": np.mean([s.pearson for s in scores]),
                "mae_norm": np.mean([s.mae_norm for s in scores]),
            }
        )
    return rows


def _check_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if set(submission[ROW_ID]) != set(sample[ROW_ID]) or len(submission) != len(sample):
        raise ValueError("submission row_ids do not match sample_submission.csv")
    if not np.isfinite(submission["predicted_score"]).all():
        raise ValueError("submission has non-finite predictions")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    train = load_split(args.data_dir, "train")
    test = load_split(args.data_dir, "test")
    lemmatizer = Lemmatizer()
    started = time.perf_counter()
    features_train = build_features(train, lemmatizer)
    features_test = build_features(test, lemmatizer)
    print(f"features built in {time.perf_counter() - started:.1f}s")
    features_train.assign(**{TARGET: train[TARGET]}).to_csv(args.out_dir / "features_train.csv", index=False)
    features_test.to_csv(args.out_dir / "features_test.csv", index=False)

    X_train = features_train[list(FEATURE_COLUMNS)]
    X_test = features_test[list(FEATURE_COLUMNS)]
    y = train[TARGET].to_numpy()
    groups = train[CASE_ID]

    results = [
        run_cv("block_prior", block_prior, X_train, y, groups, seeds, args.folds),
        run_cv("gbm", gbm, X_train, y, groups, seeds, args.folds),
    ]
    summary = pd.DataFrame([row for r in results for row in _summary(r)])
    print(f"\n{args.folds}-fold GroupKFold by case x {len(seeds)} seeds")
    print(summary.round(4).to_string(index=False))

    gbm_result = results[1]
    pd.DataFrame(
        {
            ROW_ID: train[ROW_ID],
            TARGET: y,
            "oof_raw": gbm_result.oof_raw.mean(axis=0),
            "oof_rescaled": gbm_result.oof_rescaled.mean(axis=0),
        }
    ).to_csv(args.out_dir / "oof_gbm.csv", index=False)

    importance = cv_permutation_importance(
        gbm, X_train, y, groups, seed=0, n_splits=args.folds, n_repeats=args.importance_repeats
    )
    importance.to_csv(args.out_dir / "importance_gbm.csv", index=False)
    print("\npermutation importance (held-out MSE increase, seed 0)")
    print(importance.round(4).to_string(index=False))

    submission = pd.DataFrame(
        {ROW_ID: test[ROW_ID], "predicted_score": fit_predict(gbm, X_train, y, X_test, seeds)}
    )
    _check_submission(submission, load_sample_submission(args.data_dir))
    path = args.out_dir / f"submission_gbm_{datetime.now():%Y%m%d_%H%M%S}.csv"
    submission.to_csv(path, index=False)
    print(f"\nsubmission written to {path}")
