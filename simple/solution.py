"""LegalEval Extract: predict expert scores for extracted court-decision blocks.

Model: 0.6 x gradient boosting + 0.4 x ridge regression on simple text-overlap features.
Validation: 5-fold GroupKFold by case, scored with the competition formula.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

WINDOW_SIZE = 50
WINDOW_STEP = 25
GBM_WEIGHT = 0.6
GBM_TREES = 150
RIDGE_ALPHA = 5.0
N_FOLDS = 5
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "ext_chars",
    "ext_to_src_words",
    "copy_ratio",
    "novel_words",
    "trigram_copy_ratio",
    "tfidf_max",
    "tfidf_top3_mean",
    "best_pos",
]
FEATURES = ["block_name"] + NUMERIC_FEATURES


# ---------- features ----------

def words(text):
    return re.findall(r"\w+", text.lower())


def trigrams(tokens):
    return set(zip(tokens, tokens[1:], tokens[2:]))


def share(part, whole):
    if whole == 0:
        return np.nan
    return part / whole


def row_features(source_text, extracted_text):
    src = words(source_text)
    ext = words(extracted_text)
    src_set = set(src)
    ext_set = set(ext)

    windows = []
    for start in range(0, max(1, len(src) - WINDOW_STEP), WINDOW_STEP):
        windows.append(" ".join(src[start : start + WINDOW_SIZE]))
    tfidf = TfidfVectorizer().fit(windows)
    similarities = cosine_similarity(tfidf.transform([" ".join(ext)]), tfidf.transform(windows))[0]

    return {
        "ext_chars": len(extracted_text),
        "ext_to_src_words": share(len(ext), len(src)),
        "copy_ratio": share(len(ext_set & src_set), len(ext_set)),
        "novel_words": len(ext_set - src_set),
        "trigram_copy_ratio": share(len(trigrams(ext) & trigrams(src)), len(trigrams(ext))),
        "tfidf_max": similarities.max(),
        "tfidf_top3_mean": np.sort(similarities)[-3:].mean(),
        "best_pos": share(similarities.argmax(), len(similarities) - 1),
    }


def make_features(df, block_names):
    rows = [row_features(s, e) for s, e in zip(df["source_text"], df["extracted_text"])]
    features = pd.DataFrame(rows, index=df.index)
    features["block_name"] = pd.Categorical(df["block_name"], categories=block_names)
    return features[FEATURES]


# ---------- model ----------

def make_gbm():
    return HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=GBM_TREES,
        max_leaf_nodes=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def make_ridge():
    preprocess = ColumnTransformer(
        [
            ("numbers", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
            ("block", OneHotEncoder(handle_unknown="ignore"), ["block_name"]),
        ]
    )
    return make_pipeline(preprocess, Ridge(alpha=RIDGE_ALPHA))


def fit_and_predict(X_train, y_train, X_new):
    gbm = make_gbm().fit(X_train, y_train)
    ridge = make_ridge().fit(X_train, y_train)
    return GBM_WEIGHT * gbm.predict(X_new) + (1 - GBM_WEIGHT) * ridge.predict(X_new)


def rescale(predictions, target_mean, target_std):
    stretched = (predictions - predictions.mean()) / predictions.std() * target_std + target_mean
    return np.clip(stretched, 1, 5)


# ---------- metric ----------

def competition_score(y_true, y_pred):
    mae_part = 1 - np.mean(np.abs(y_true - y_pred)) / 4
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    covariance = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    ccc = 2 * covariance / (y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2)
    total = 0.15 * mae_part + 0.55 * ccc + 0.30 * pearson
    return {"total": total, "ccc": ccc, "pearson": pearson, "mae_part": mae_part}


# ---------- main ----------

def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    block_names = sorted(set(train["block_name"]) | set(test["block_name"]))
    print("building features...")
    X_train = make_features(train, block_names)
    X_test = make_features(test, block_names)
    y = train["expert_score_mean"].to_numpy()

    print(f"{N_FOLDS}-fold cross-validation, grouped by case:")
    oof = np.zeros(len(train))
    folds = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (fit_rows, valid_rows) in enumerate(folds.split(X_train, y, groups=train["id_sud_resh"]), 1):
        oof[valid_rows] = fit_and_predict(X_train.iloc[fit_rows], y[fit_rows], X_train.iloc[valid_rows])
        print(f"  fold {fold} done")
    oof = rescale(oof, y.mean(), y.std())
    for name, value in competition_score(y, oof).items():
        print(f"  {name}: {value:.4f}")

    test_predictions = fit_and_predict(X_train, y, X_test)
    submission = pd.DataFrame({"row_id": test["row_id"], "predicted_score": rescale(test_predictions, y.mean(), y.std())})

    assert list(submission["row_id"]) == list(sample["row_id"]), "row_ids must match sample_submission.csv"
    assert submission["predicted_score"].between(1, 5).all(), "scores must be between 1 and 5"

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"saved {OUTPUT_DIR / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
