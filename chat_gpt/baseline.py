#!/usr/bin/env python3
"""Grouped CV baseline for predicting Russian judicial extraction expert scores."""
import argparse
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
NUM = re.compile(r"\d+", flags=re.UNICODE)
NUMERIC_COLUMNS = [
    "log_extract_chars", "log_source_chars", "log_extract_words",
    "log_source_words", "length_ratio", "token_coverage", "bigram_coverage",
    "unique_token_ratio", "source_token_jaccard", "numbers_supported",
    "number_count", "exact_substring"
]


@lru_cache(maxsize=1024)
def source_stats(s: str):
    toks = TOKEN.findall(s.lower())
    word_set = set(toks)
    bigrams = set(zip(toks, toks[1:]))
    numbers = set(NUM.findall(s))
    return word_set, bigrams, numbers, len(toks)


def make_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row features; no training labels are used."""
    values = []
    for src, ext in zip(df["source_text"], df["extracted_text"]):
        src = src or ""
        ext = ext or ""
        src_words, src_bigrams, src_numbers, n_src_words = source_stats(src)
        tokens = TOKEN.findall(ext.lower())
        ext_words = set(tokens)
        ext_bigrams = set(zip(tokens, tokens[1:]))
        nums = set(NUM.findall(ext))

        shared = len(ext_words & src_words)
        values.append([
            np.log1p(len(ext)), np.log1p(len(src)),
            np.log1p(len(tokens)), np.log1p(n_src_words),
            len(ext) / max(1, len(src)),
            shared / max(1, len(ext_words)),
            len(ext_bigrams & src_bigrams) / max(1, len(ext_bigrams)),
            len(ext_words) / max(1, len(tokens)),
            shared / max(1, len(ext_words | src_words)),
            len(nums & src_numbers) / max(1, len(nums)) if nums else 1.0,
            float(len(nums)),
            float(bool(ext.strip()) and ext.strip().casefold() in src.casefold()),
        ])
    return pd.DataFrame(values, columns=NUMERIC_COLUMNS, index=df.index)


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    for df in (train, test):
        for col in ("source_text", "extracted_text", "block_name"):
            df[col] = df[col].fillna("").astype(str)
    required = ["row_id", "id_sud_resh", "block_name", "source_text", "extracted_text"]
    assert set(required + ["expert_score_mean"]).issubset(train.columns)
    assert set(required).issubset(test.columns)
    assert list(sample.columns) == ["row_id", "predicted_score"]
    assert len(sample) == len(test)
    assert sample["row_id"].equals(test["row_id"]), "sample rows not aligned with test!"
    assert train["row_id"].is_unique and test["row_id"].is_unique
    assert not set(train["id_sud_resh"]) & set(test["id_sud_resh"])
    assert train["expert_score_mean"].between(1, 5).all()
    return train, test, sample


def build_inputs(train, test, mode="enhanced"):
    train_x = train[["extracted_text", "block_name"]].copy()
    test_x = test[["extracted_text", "block_name"]].copy()
    if mode == "enhanced":
        for df, x in [(train, train_x), (test, test_x)]:
            z = make_numeric_features(df)
            for col in NUMERIC_COLUMNS:
                x[col] = z[col].to_numpy()
    return train_x, test_x


def build_model(mode="enhanced", alpha=10.0):
    transforms = [
        ("word", TfidfVectorizer(ngram_range=(1, 2),
                                 min_df=2, max_features=30000),
         "extracted_text"),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                 min_df=2, max_features=40000),
         "extracted_text"),
        ("block", OneHotEncoder(handle_unknown="ignore"), ["block_name"]),
    ]
    weights = {"word": 1.0, "char": 1.0, "block": 1.0}
    if mode == "enhanced":
        transforms.append(("numbers", StandardScaler(), NUMERIC_COLUMNS))
        weights["numbers"] = 0.30
    return Pipeline([
        ("features", ColumnTransformer(transforms,
                                       transformer_weights=weights)),
        ("model", Ridge(alpha=alpha)),
    ])


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    mae = float(np.mean(np.abs(y - p)))
    if np.std(y) < 1e-12 or np.std(p) < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(y, p)[0, 1])
    cov = float(np.mean((y - y.mean()) * (p - p.mean())))
    ccc = 2.0 * cov / (np.var(y) + np.var(p) + (y.mean() - p.mean()) ** 2)
    return {"MAE": mae, "Pearson": r, "CCC": float(ccc)}


def oof_predict(model_factory, x, y, groups, n_splits=5):
    predictions = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (idx_train, idx_valid) in enumerate(gkf.split(x, y, groups), 1):
        model = model_factory()
        model.fit(x.iloc[idx_train], y[idx_train])
        predictions[idx_valid] = np.clip(model.predict(x.iloc[idx_valid]), 1, 5)
        print(f"Fold {fold}/{n_splits}: {len(idx_valid)} validation rows", flush=True)
    assert np.isfinite(predictions).all()
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True, parents=True)

    train, test, sample = load_data()
    x, xt = build_inputs(train, test, mode=args.mode)
    y = train["expert_score_mean"].to_numpy()
    groups = train["id_sud_resh"].to_numpy()
    factory = lambda: build_model(mode=args.mode, alpha=args.alpha)

    oof = oof_predict(factory, x, y, groups, args.folds)
    print(f"{args.mode} grouped OOF:", metrics(y, oof), flush=True)
    pd.DataFrame({
        "row_id": train["row_id"],
        "expert_score_mean": y,
        "predicted_score": oof,
    }).to_csv(OUTPUTS / f"oof_{args.mode}.csv", index=False)

    final = factory()
    final.fit(x, y)
    sample["predicted_score"] = np.clip(final.predict(xt), 1, 5)
    assert sample["predicted_score"].notna().all()
    assert sample["predicted_score"].between(1, 5).all()
    path = OUTPUTS / f"submission_{args.mode}.csv"
    sample.to_csv(path, index=False)
    print("Wrote:", path.resolve())


if __name__ == "__main__":
    main()
