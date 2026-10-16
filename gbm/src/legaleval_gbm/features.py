from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from legaleval_gbm.data import BLOCK, BLOCKS, CASE_ID, EXTRACTED, ROW_ID, SOURCE
from legaleval_gbm.text import Lemmatizer, is_number, trigrams, windows

WINDOW_SIZE = 50
WINDOW_STRIDE = 25
TOP_K = 3

FEATURE_COLUMNS = (
    BLOCK,
    "ext_chars",
    "ext_words",
    "src_words",
    "ext_to_src_words",
    "copy_ratio",
    "novel_words",
    "trigram_copy_ratio",
    "num_count",
    "num_copy_ratio",
    "tfidf_max",
    "tfidf_topk_mean",
    "bm25_max_per_term",
    "best_window_pos",
)


@dataclass(frozen=True)
class SourceIndex:
    n_words: int
    vocabulary: frozenset[str]
    numbers: frozenset[str]
    trigrams: frozenset[tuple[str, str, str]]
    n_windows: int
    tfidf: TfidfVectorizer
    window_vectors: csr_matrix
    bm25: BM25Okapi

    @classmethod
    def build(cls, tokens: list[str]) -> "SourceIndex":
        chunks = windows(tokens, WINDOW_SIZE, WINDOW_STRIDE)
        tfidf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
        return cls(
            n_words=len(tokens),
            vocabulary=frozenset(tokens),
            numbers=frozenset(t for t in tokens if is_number(t)),
            trigrams=frozenset(trigrams(tokens)),
            n_windows=len(chunks),
            tfidf=tfidf,
            window_vectors=tfidf.fit_transform([" ".join(c) for c in chunks]),
            bm25=BM25Okapi(chunks),
        )


def _share(part: float, whole: int) -> float:
    return part / whole if whole else np.nan


def row_features(extracted: list[str], extracted_chars: int, source: SourceIndex) -> dict[str, float]:
    unique = set(extracted)
    numbers = {t for t in unique if is_number(t)}
    grams = trigrams(extracted)
    query = source.tfidf.transform([" ".join(extracted)])
    similarities = linear_kernel(query, source.window_vectors).ravel()
    top = np.sort(similarities)[::-1][:TOP_K]
    best = int(np.argmax(similarities))
    bm25 = source.bm25.get_scores(list(unique))
    return {
        "ext_chars": extracted_chars,
        "ext_words": len(extracted),
        "src_words": source.n_words,
        "ext_to_src_words": _share(len(extracted), source.n_words),
        "copy_ratio": _share(len(unique & source.vocabulary), len(unique)),
        "novel_words": len(unique - source.vocabulary),
        "trigram_copy_ratio": _share(len(grams & source.trigrams), len(grams)),
        "num_count": len(numbers),
        "num_copy_ratio": _share(len(numbers & source.numbers), len(numbers)),
        "tfidf_max": float(top[0]),
        "tfidf_topk_mean": float(top.mean()),
        "bm25_max_per_term": _share(float(bm25.max()), len(unique)),
        "best_window_pos": best / (source.n_windows - 1) if source.n_windows > 1 else np.nan,
    }


def build_features(df: pd.DataFrame, lemmatizer: Lemmatizer) -> pd.DataFrame:
    indexes = {
        case: SourceIndex.build(lemmatizer.lemmas(text))
        for case, text in df.drop_duplicates(CASE_ID)[[CASE_ID, SOURCE]].itertuples(index=False)
    }
    rows = [
        row_features(lemmatizer.lemmas(text), len(text), indexes[case])
        for case, text in df[[CASE_ID, EXTRACTED]].itertuples(index=False)
    ]
    features = pd.concat([df[[ROW_ID, CASE_ID]], pd.DataFrame(rows, index=df.index)], axis=1)
    features.insert(2, BLOCK, pd.Categorical(df[BLOCK], categories=BLOCKS))
    return features
