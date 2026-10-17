#!/usr/bin/env python3
"""Inspect OOF errors globally and by extraction block."""
import argparse
import pandas as pd
from baseline import DATA, OUTPUTS, metrics

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=str(OUTPUTS / "oof_enhanced.csv"))
args = parser.parse_args()
train = pd.read_csv(DATA / "train.csv")
oof = pd.read_csv(args.file)
assert oof["row_id"].equals(train["row_id"])
d = train.merge(oof[["row_id", "predicted_score"]],
                on="row_id", validate="one_to_one")
print("Overall:", metrics(d["expert_score_mean"], d["predicted_score"]))
print("\nPer block:")
scores = []
for block, part in d.groupby("block_name"):
    scores.append({"block_name": block, **metrics(
        part["expert_score_mean"], part["predicted_score"])})
print(pd.DataFrame(scores).sort_values("MAE", ascending=False).to_string(
    index=False, float_format=lambda n: f"{n:.3f}"))
d["abs_error"] = (d["expert_score_mean"] - d["predicted_score"]).abs()
print("\nTop 10 errors:")
print(d.nlargest(10, "abs_error")[[
    "row_id", "expert_score_mean", "predicted_score", "abs_error",
]].to_string(index=False, float_format=lambda n: f"{n:.3f}"))
