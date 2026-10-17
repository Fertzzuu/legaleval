#!/usr/bin/env python3
"""Verify a competition submission has correct shape, order, types and score range."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from baseline import DATA

parser = argparse.ArgumentParser()
parser.add_argument("submission", type=Path)
args = parser.parse_args()
sample = pd.read_csv(DATA / "sample_submission.csv")
test = pd.read_csv(DATA / "test.csv", usecols=["row_id"])
submission = pd.read_csv(args.submission)
assert list(submission.columns) == ["row_id", "predicted_score"]
assert len(submission) == len(sample) == len(test)
assert submission["row_id"].equals(sample["row_id"])
assert submission["row_id"].equals(test["row_id"])
assert submission["row_id"].is_unique
assert pd.api.types.is_numeric_dtype(submission["predicted_score"])
values = submission["predicted_score"].to_numpy(dtype=float)
assert np.isfinite(values).all()
assert ((1 <= values) & (values <= 5)).all()
print(f"VALID: {args.submission} | rows={len(submission)} | "
      f"min={values.min():.3f} | max={values.max():.3f}")
