import math

import pandas as pd
import pytest

from legaleval_gbm.data import BLOCKS
from legaleval_gbm.features import (
    CROSS_BLOCK_COLUMNS,
    LEXICAL_COLUMNS,
    SourceIndex,
    build_features,
    row_features,
)
from legaleval_gbm.text import Lemmatizer, windows


@pytest.fixture(scope="module")
def lemmatizer() -> Lemmatizer:
    return Lemmatizer()


def test_lemmas_reduce_inflected_forms(lemmatizer):
    assert lemmatizer.lemmas("Судом, истца!") == ["суд", "истец"]


def test_windows_cover_the_tail():
    tokens = [str(i) for i in range(110)]
    chunks = windows(tokens, size=50, stride=25)
    assert chunks[0] == tokens[:50]
    assert chunks[-1][-1] == "109"


def test_windows_keep_short_text_whole():
    assert windows(["a", "b"], size=50, stride=25) == [["a", "b"]]


def test_copied_extraction_scores_full_overlap(lemmatizer):
    text = "Истец просит взыскать с ответчика 100 рублей по договору 5"
    source = SourceIndex.build(lemmatizer.lemmas(text))
    features = row_features(lemmatizer.lemmas("взыскать с ответчика 100 рублей"), 30, source)
    assert features["copy_ratio"] == pytest.approx(1.0)
    assert features["trigram_copy_ratio"] == pytest.approx(1.0)
    assert features["num_copy_ratio"] == pytest.approx(1.0)
    assert features["novel_words"] == 0


def test_invented_extraction_scores_low_overlap(lemmatizer):
    source = SourceIndex.build(lemmatizer.lemmas("Истец просит взыскать с ответчика 100 рублей"))
    features = row_features(lemmatizer.lemmas("суд отказал в иске 999"), 22, source)
    assert features["copy_ratio"] == pytest.approx(0.0)
    assert features["num_copy_ratio"] == pytest.approx(0.0)
    assert features["tfidf_max"] == pytest.approx(0.0)


def test_undefined_ratios_are_nan(lemmatizer):
    source = SourceIndex.build(lemmatizer.lemmas("Истец просит взыскать долг"))
    features = row_features(lemmatizer.lemmas("долг взыскать"), 13, source)
    assert math.isnan(features["num_copy_ratio"])
    assert math.isnan(features["trigram_copy_ratio"])
    assert math.isnan(features["best_window_pos"])


def test_build_features_shape_and_block_categories(lemmatizer):
    df = pd.DataFrame(
        {
            "row_id": ["c1|" + BLOCKS[0], "c1|" + BLOCKS[6]],
            "id_sud_resh": ["c1", "c1"],
            "block_name": [BLOCKS[0], BLOCKS[6]],
            "extracted_text": ["взыскать долг", "иск удовлетворить"],
            "source_text": ["Истец просит взыскать долг. Иск удовлетворить."] * 2,
        }
    )
    features = build_features(df, lemmatizer)
    assert list(features.columns) == ["row_id", "id_sud_resh", *LEXICAL_COLUMNS, *CROSS_BLOCK_COLUMNS]
    assert list(features["block_name"].cat.categories) == list(BLOCKS)


def _case_frame(case_ids: list[str], extracted: list[str], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"{c}|{BLOCKS[i]}" for i, c in enumerate(case_ids)],
            "id_sud_resh": case_ids,
            "block_name": [BLOCKS[i] for i in range(len(case_ids))],
            "extracted_text": extracted,
            "source_text": [source] * len(case_ids),
        }
    )


def test_cross_block_jaccard_flags_duplicated_extractions(lemmatizer):
    extracted = ["взыскать долг", "взыскать долг", "иск"]
    df = _case_frame(["c1", "c1", "c2"], extracted, "Истец просит взыскать долг")
    features = build_features(df, lemmatizer)
    assert features["cross_block_jaccard_max"].iloc[0] == pytest.approx(1.0)
    assert features["cross_block_jaccard_max"].iloc[1] == pytest.approx(1.0)
    assert math.isnan(features["cross_block_jaccard_max"].iloc[2])
