import numpy as np
import pandas as pd
from legaleval_gbm.cv import CvResult
from legaleval_gbm.data import ROW_ID


def summarize(result: CvResult) -> list[dict[str, object]]:
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


def check_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if set(submission[ROW_ID]) != set(sample[ROW_ID]) or len(submission) != len(sample):
        raise ValueError("submission row_ids do not match sample_submission.csv")
    if not np.isfinite(submission["predicted_score"]).all():
        raise ValueError("submission has non-finite predictions")
