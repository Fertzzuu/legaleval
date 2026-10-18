import math

import numpy as np
from legaleval_gbm.data import BLOCKS
from legaleval_gbm.features import build_features

from legaleval_agent.judge_model import JudgeModel, Range, reasons

RANGES = {
    "ext_chars": Range(100, 250, 600),
    "copy_ratio": Range(0.8, 0.95, 1.0),
    "trigram_copy_ratio": Range(0.4, 0.7, 0.9),
    "novel_words": Range(0, 2, 6),
}


def test_reasons_flag_short_and_invented_extractions():
    values = {"ext_chars": 40.0, "copy_ratio": 0.5, "trigram_copy_ratio": 0.1, "novel_words": 12.0}
    found = reasons(values, BLOCKS[0], RANGES)
    assert len(found) == 4
    assert found[0].startswith("shorter than 90%")
    assert "50% of its words" in found[1]


def test_reasons_are_empty_for_typical_extractions():
    values = {"ext_chars": 250.0, "copy_ratio": 0.95, "trigram_copy_ratio": math.nan, "novel_words": 2.0}
    assert reasons(values, BLOCKS[0], RANGES) == []


def test_frozen_rescale_scores_one_extraction_the_same_alone_or_in_a_batch(
    judge, synthetic_train, lemmatizer
):
    batch = synthetic_train.head(7)
    together = judge.score_features(build_features(batch, lemmatizer))
    alone = [judge.score_features(build_features(batch.iloc[[i]], lemmatizer))[0] for i in range(len(batch))]
    assert np.allclose(together, alone)
    assert ((1 <= together) & (together <= 5)).all()


def test_judge_returns_a_verdict(judge, synthetic_train, lemmatizer):
    row = synthetic_train.iloc[0]
    verdict = judge.judge(row["source_text"], row["block_name"], row["extracted_text"], lemmatizer)
    assert 1 <= verdict.score <= 5
    assert verdict.typical_score_for_block == round(judge.block_scores[row["block_name"]], 2)


def test_save_and_load_round_trip(judge, tmp_path, synthetic_train, lemmatizer):
    path = tmp_path / "judge.joblib"
    judge.save(path)
    loaded = JudgeModel.load(path)
    features = build_features(synthetic_train.head(3), lemmatizer)
    assert np.allclose(loaded.score_features(features), judge.score_features(features))
