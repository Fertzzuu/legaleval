#!/usr/bin/env python3
"""Semantic embeddings + grouped Ridge CV + optional blend with enhanced baseline.

Run baseline.py --mode enhanced before this script.
"""
import argparse
import hashlib
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baseline import (
    ROOT, DATA, OUTPUTS, NUMERIC_COLUMNS, load_data,
    make_numeric_features, metrics, oof_predict,
)

MODEL_ID = "intfloat/multilingual-e5-small"
CACHE = ROOT / "cache"
WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


@lru_cache(maxsize=1024)
def windows(text: str, size=1050, step=850):
    # Character chunks stay below the encoder's 512-token limit in most cases.
    # The encoder still applies its own token-limit truncation when necessary.
    pieces = []
    for i in range(0, len(text), step):
        snippet = text[i:i + size].strip()
        if snippet:
            pieces.append(snippet)
        if i + size >= len(text):
            break
    return pieces or [""]


def source_passages(src, ext, top_k=2):
    terms = set(WORD_RE.findall(ext.lower()))
    pieces = windows(src)
    ranked = sorted(
        pieces,
        key=lambda s: len(terms & set(WORD_RE.findall(s.lower()))),
        reverse=True,
    )
    chosen = ranked[:top_k]
    return chosen + [""] * (top_k - len(chosen))


def signature_for(df, model):
    h = hashlib.sha256(model.encode())
    for row in df[["row_id", "extracted_text", "source_text"]].itertuples(index=False, name=None):
        for field in row:
            h.update(str(field).encode("utf-8"))
            h.update(b"\x00")
    return h.hexdigest()


def compute_embeddings(df, batch_size=32):
    """Unsupervised pretrained embeddings of each extraction and its source passages."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading encoder: {MODEL_ID}", flush=True)
    model = SentenceTransformer(MODEL_ID)

    queries = ["query: " + s for s in df["extracted_text"]]
    chosen = [source_passages(src, ext) for src, ext in
              zip(df["source_text"], df["extracted_text"])]

    # Deduplicate source passages to reduce unnecessary model inference.
    passages = list(dict.fromkeys("passage: " + piece
                                  for pair in chosen for piece in pair))
    index = {p: i for i, p in enumerate(passages)}
    pairs = np.array([[index["passage: " + p] for p in pair] for pair in chosen])

    print(f"Encoding {len(queries)} extractions and {len(passages)} "
          "distinct source passages...", flush=True)
    q = model.encode(queries, batch_size=batch_size,
                     normalize_embeddings=True, show_progress_bar=True)
    p = model.encode(passages, batch_size=batch_size,
                     normalize_embeddings=True, show_progress_bar=True)
    q, p = np.asarray(q, dtype=np.float32), np.asarray(p, dtype=np.float32)

    selected = p[pairs]                     # (n, 2, embedding_dim)
    similarities = np.einsum("nd,nkd->nk", q, selected)
    context = selected.mean(axis=1)         # (n, embedding_dim)
    sim = np.column_stack([
        similarities[:, 0], similarities[:, 1],
        similarities.max(axis=1), similarities.mean(axis=1),
        similarities.max(axis=1) - similarities.min(axis=1),
    ])
    return np.hstack([q, context, sim]).astype(np.float32)


def build_frame(train, test, batch_size=32):
    CACHE.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([train, test], ignore_index=True)
    signature = signature_for(combined, MODEL_ID + ":top2_1050_850")
    path = CACHE / "embeddings_e5small.npz"
    if path.exists():
        with np.load(path) as z:
            if str(z["signature"]) == signature:
                arr = z["features"]
                print("Reusing cached embeddings:", path, flush=True)
            else:
                arr = compute_embeddings(combined, batch_size=batch_size)
                np.savez_compressed(path, features=arr, signature=signature)
    else:
        arr = compute_embeddings(combined, batch_size=batch_size)
        np.savez_compressed(path, features=arr, signature=signature)

    em_cols = [f"emb_{i:03d}" for i in range(arr.shape[1])]
    em = pd.DataFrame(arr, columns=em_cols)
    nums = make_numeric_features(combined).reset_index(drop=True)
    features = pd.concat(
        [combined[["block_name"]].reset_index(drop=True), nums, em],
        axis=1,
    )
    return features.iloc[:len(train)].copy(), features.iloc[len(train):].copy(), em_cols


def make_model(em_cols, alpha=80.0):
    return Pipeline([
        ("features", ColumnTransformer([
            ("emb", StandardScaler(), em_cols),
            ("numerics", StandardScaler(), NUMERIC_COLUMNS),
            ("block", OneHotEncoder(handle_unknown="ignore"), ["block_name"]),
        ], transformer_weights={"emb": 0.25, "numerics": 0.35, "block": 1.0})),
        ("model", Ridge(alpha=alpha)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=80.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--blend-weight", type=float, default=0.5,
                        help="Weight on semantic model, remaining weight on baseline")
    args = parser.parse_args()
    assert 0 <= args.blend_weight <= 1

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    train, test, sample = load_data()
    x, xt, em_cols = build_frame(train, test, args.batch_size)
    y = train["expert_score_mean"].to_numpy()
    factory = lambda: make_model(em_cols, alpha=args.alpha)

    oof = oof_predict(factory, x, y, train["id_sud_resh"].to_numpy())
    print("Embedding grouped OOF:", metrics(y, oof), flush=True)
    pd.DataFrame({
        "row_id": train["row_id"],
        "expert_score_mean": y,
        "predicted_score": oof,
    }).to_csv(OUTPUTS / "oof_embeddings.csv", index=False)

    fitted = factory()
    fitted.fit(x, y)
    sample["predicted_score"] = np.clip(fitted.predict(xt), 1, 5)
    sample.to_csv(OUTPUTS / "submission_embeddings.csv", index=False)

    baseline_oof_path = OUTPUTS / "oof_enhanced.csv"
    baseline_test_path = OUTPUTS / "submission_enhanced.csv"
    if baseline_oof_path.exists() and baseline_test_path.exists():
        base_oof = pd.read_csv(baseline_oof_path)
        base_test = pd.read_csv(baseline_test_path)
        assert base_oof["row_id"].equals(train["row_id"])
        assert base_test["row_id"].equals(test["row_id"])
        w = args.blend_weight
        blended_oof = np.clip(
            (1 - w) * base_oof["predicted_score"].to_numpy() + w * oof, 1, 5)
        blended_test = np.clip(
            (1 - w) * base_test["predicted_score"].to_numpy()
            + w * sample["predicted_score"].to_numpy(), 1, 5)
        print(f"{w:.0%} embeddings blend OOF:",
              metrics(y, blended_oof), flush=True)
        blend = pd.DataFrame({
            "row_id": sample["row_id"],
            "predicted_score": blended_test,
        })
        blend.to_csv(OUTPUTS / "submission_blended.csv", index=False)
        print("Wrote:", OUTPUTS / "submission_blended.csv")
    print("Wrote:", OUTPUTS / "submission_embeddings.csv")


if __name__ == "__main__":
    main()
