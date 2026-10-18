import json

import pandas as pd
import pytest
from legaleval_gbm.data import BLOCKS

from legaleval_qwen.judge import JudgmentStore, cases, grades_frame
from legaleval_qwen.logit import LogitJudgment, answer_opening, answer_step, judge_case_logits


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"c1|{BLOCKS[0]}", f"c1|{BLOCKS[6]}"],
            "id_sud_resh": ["c1", "c1"],
            "block_name": [BLOCKS[0], BLOCKS[6]],
            "extracted_text": ["взыскать долг", "иск удовлетворить"],
            "source_text": ["решение один", "решение один"],
        }
    )


def _fake_scorer(chat, blocks):
    probabilities = {1: [0.0, 0.0, 0.0, 0.5, 0.5], 2: [0.0, 0.0, 1.0, 0.0, 0.0]}
    return {b: probabilities[b] for b in blocks}, {b: 0.99 for b in blocks}, 123


def test_forced_answer_is_the_models_natural_json():
    text = answer_opening(1) + answer_step(5, 2) + answer_step(3, 3) + "4\n}"
    assert text == '{\n  "1": 5,\n  "2": 3,\n  "3": 4\n}'
    assert json.loads(text) == {"1": 5, "2": 3, "3": 4}


def test_expected_grade_is_probability_weighted():
    judgment = judge_case_logits(next(cases(_frame())), "fake", _fake_scorer)
    assert judgment.expected_grades() == pytest.approx({1: 4.5, 2: 3.0})
    assert judgment.prompt_tokens == 123


def test_store_round_trip_restores_integer_block_keys(tmp_path):
    store = JudgmentStore(tmp_path / "l.jsonl", LogitJudgment.from_record)
    store.append(judge_case_logits(next(cases(_frame())), "fake", _fake_scorer))
    loaded = store.load()["c1"]
    assert set(loaded.probabilities) == {1, 2}
    assert loaded.grade_mass == {1: 0.99, 2: 0.99}


def test_expected_grades_feed_grades_frame():
    df = _frame()
    judgment = judge_case_logits(next(cases(df)), "fake", _fake_scorer)
    grades = grades_frame(df, {"c1": judgment.expected_grades()}).set_index("row_id")["llm_grade"]
    assert grades[f"c1|{BLOCKS[0]}"] == pytest.approx(4.5)
    assert grades[f"c1|{BLOCKS[6]}"] == pytest.approx(3.0)
