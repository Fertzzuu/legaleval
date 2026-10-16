from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from legaleval_gbm.data import BLOCK, BLOCKS, CASE_ID, EXTRACTED, ROW_ID, SOURCE
from legaleval_gbm.embeddings import PASSAGE_PREFIX, QUERY_PREFIX, CachedEmbedder
from legaleval_gbm.text import Lemmatizer, is_number, trigrams, windows

WINDOW_SIZE = 50
WINDOW_STRIDE = 25
TOP_K = 3

LEXICAL_COLUMNS = (
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
CROSS_BLOCK_COLUMNS = ("cross_block_jaccard_max",)
EMBEDDING_COLUMNS = ("emb_max_sim", "emb_topk_mean", "emb_best_pos", "emb_cross_block_max")

FEATURE_SETS = {
    "lexical": LEXICAL_COLUMNS,
    "lexical+cross_block": LEXICAL_COLUMNS + CROSS_BLOCK_COLUMNS,
    "lexical+embedding": LEXICAL_COLUMNS + EMBEDDING_COLUMNS,
    "all": LEXICAL_COLUMNS + CROSS_BLOCK_COLUMNS + EMBEDDING_COLUMNS,
}


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


def _max_off_diagonal(similarity: np.ndarray) -> np.ndarray:
    if len(similarity) < 2:
        return np.full(len(similarity), np.nan)
    masked = similarity.astype(float)
    np.fill_diagonal(masked, -np.inf)
    return masked.max(axis=1)


def _jaccard_matrix(sets: list[set[str]]) -> np.ndarray:
    return np.array([[_share(len(a & b), len(a | b)) for b in sets] for a in sets])


def _per_case(cases: pd.Series, compute: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    values = np.empty(len(cases))
    for positions in cases.groupby(cases, sort=False).indices.values():
        values[positions] = compute(positions)
    return values


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
    extracted = [lemmatizer.lemmas(text) for text in df[EXTRACTED]]
    rows = [
        row_features(lemmas, len(text), indexes[case])
        for lemmas, text, case in zip(extracted, df[EXTRACTED], df[CASE_ID], strict=True)
    ]
    features = pd.concat([df[[ROW_ID, CASE_ID]], pd.DataFrame(rows, index=df.index)], axis=1)
    features.insert(2, BLOCK, pd.Categorical(df[BLOCK], categories=BLOCKS))
    features["cross_block_jaccard_max"] = _per_case(
        df[CASE_ID], lambda idx: _max_off_diagonal(_jaccard_matrix([set(extracted[i]) for i in idx]))
    )
    return features


def embedding_features(df: pd.DataFrame, embedder: CachedEmbedder) -> pd.DataFrame:
    sources = df.drop_duplicates(CASE_ID).set_index(CASE_ID)[SOURCE]
    passages = {
        case: [PASSAGE_PREFIX + " ".join(w) for w in windows(text.split(), WINDOW_SIZE, WINDOW_STRIDE)]
        for case, text in sources.items()
    }
    window_vectors = embedder.embed([p for chunk in passages.values() for p in chunk])
    bounds = np.cumsum([0, *(len(chunk) for chunk in passages.values())])
    by_case = {
        case: window_vectors[start:end]
        for case, start, end in zip(passages, bounds[:-1], bounds[1:], strict=True)
    }
    queries = embedder.embed([QUERY_PREFIX + text for text in df[EXTRACTED]])
    similarities = [by_case[case] @ q for case, q in zip(df[CASE_ID], queries, strict=True)]
    top = [np.sort(s)[::-1][:TOP_K] for s in similarities]
    features = pd.DataFrame(
        {
            "emb_max_sim": [t[0] for t in top],
            "emb_topk_mean": [t.mean() for t in top],
            "emb_best_pos": [s.argmax() / (len(s) - 1) if len(s) > 1 else np.nan for s in similarities],
        },
        index=df.index,
    )
    features["emb_cross_block_max"] = _per_case(
        df[CASE_ID], lambda idx: _max_off_diagonal(queries[idx] @ queries[idx].T)
    )
    return features
