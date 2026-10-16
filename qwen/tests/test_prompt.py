import pytest
from legaleval_gbm.data import BLOCKS

from legaleval_qwen.prompt import Block, JudgeParseError, messages, parse_grades

BLOCKS_2 = [Block(1, BLOCKS[0], "взыскать долг"), Block(2, BLOCKS[6], "иск удовлетворить")]


def test_parses_plain_json():
    assert parse_grades('{"1": 5, "2": 3}', BLOCKS_2) == {1: 5, 2: 3}


def test_ignores_thinking_and_surrounding_text():
    response = '<think>maybe {"1": 1, "2": 1}</think>Here you go: {"1": 4, "2": 2}'
    assert parse_grades(response, BLOCKS_2) == {1: 4, 2: 2}


@pytest.mark.parametrize(
    "response",
    [
        "no json here",
        '{"1": 5}',
        '{"1": 5, "2": 3, "3": 1}',
        '{"1": 6, "2": 3}',
        '{"1": 4.5, "2": 3}',
        '{"1": true, "2": 3}',
        '{"1": 5, "2": }',
    ],
)
def test_rejects_malformed_responses(response):
    with pytest.raises(JudgeParseError):
        parse_grades(response, BLOCKS_2)


def test_user_message_numbers_blocks_and_lists_expected_keys():
    chat = messages("Решение суда по делу", BLOCKS_2)
    assert [m["role"] for m in chat] == ["system", "user"]
    user = chat[1]["content"]
    assert f"[1] {BLOCKS[0]}" in user
    assert f"[2] {BLOCKS[6]}" in user
    assert 'Return JSON with keys "1", "2".' in user
