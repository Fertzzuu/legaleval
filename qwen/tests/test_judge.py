import numpy as np
import pandas as pd
import pytest
from legaleval_gbm.data import BLOCKS

from legaleval_qwen.calibrate import GRADE, IsotonicGrade, LinearGradeByBlock, RawGrade
from legaleval_qwen.judge import Judgment, JudgmentStore, cases, grades_frame, judge_case, text_grades


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"c1|{BLOCKS[6]}", f"c1|{BLOCKS[0]}", f"c2|{BLOCKS[0]}"],
            "id_sud_resh": ["c1", "c1", "c2"],
            "block_name": [BLOCKS[6], BLOCKS[0], BLOCKS[0]],
            "extracted_text": ["иск удовлетворить", "взыскать долг", "взыскать неустойку"],
            "source_text": ["решение один", "решение один", "решение два"],
        }
    )


def test_cases_follow_canonical_block_order():
    first = next(cases(_frame()))
    assert [b.name for b in first.blocks] == [BLOCKS[0], BLOCKS[6]]
    assert first.row_ids == {1: f"c1|{BLOCKS[0]}", 2: f"c1|{BLOCKS[6]}"}


def test_store_round_trip_keeps_latest_judgment(tmp_path):
    store = JudgmentStore(tmp_path / "j.jsonl", Judgment.from_record)
    case = next(cases(_frame()))
    store.append(judge_case(case, "fake", lambda chat: ("not json", 10)))
    store.append(judge_case(case, "fake", lambda chat: ('{"1": 5, "2": 2}', 10)))
    loaded = store.load()["c1"]
    assert loaded.grades == {1: 5, 2: 2}
    assert loaded.error is None


def test_failed_parse_is_recorded_not_raised():
    judgment = judge_case(next(cases(_frame())), "fake", lambda chat: ("{}", 3))
    assert judgment.grades is None
    assert judgment.error is not None


def test_grades_frame_maps_grades_to_rows():
    df = _frame()
    judgments = {c.case_id: judge_case(c, "fake", lambda chat: ('{"1": 4, "2": 1}', 1)) for c in cases(df)}
    judgments["c2"] = judge_case(
        next(c for c in cases(df) if c.case_id == "c2"), "f", lambda chat: ('{"1": 3}', 1)
    )
    grades = grades_frame(df, text_grades(judgments)).set_index("row_id")[GRADE]
    assert grades[f"c1|{BLOCKS[0]}"] == 4.0
    assert grades[f"c1|{BLOCKS[6]}"] == 1.0
    assert grades[f"c2|{BLOCKS[0]}"] == 3.0


def test_text_grades_skip_failed_judgments():
    df = _frame()
    judgments = {c.case_id: judge_case(c, "fake", lambda chat: ("broken", 1)) for c in cases(df)}
    assert text_grades(judgments) == {}


def test_grades_frame_refuses_missing_cases():
    df = _frame()
    with pytest.raises(ValueError, match="no valid judgment"):
        grades_frame(df, {})


def test_calibrators_fit_and_predict():
    X = pd.DataFrame(
        {
            GRADE: [5.0, 5.0, 3.0, 1.0, 4.0, 2.0],
            "block_name": [BLOCKS[0], BLOCKS[6], BLOCKS[0], BLOCKS[6], BLOCKS[0], BLOCKS[6]],
        }
    )
    y = np.array([4.5, 5.0, 3.5, 1.5, 4.0, 2.5])
    assert RawGrade().fit(X, y).predict(X) == pytest.approx(X[GRADE].to_numpy())
    isotonic = IsotonicGrade().fit(X, y).predict(X)
    assert np.all(np.diff(isotonic[np.argsort(X[GRADE].to_numpy(), kind="stable")]) >= 0)
    assert LinearGradeByBlock().fit(X, y).predict(X).shape == (6,)
