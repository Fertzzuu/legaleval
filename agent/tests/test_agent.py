import pytest
from legaleval_gbm.data import BLOCKS

from legaleval_agent.agent import LengthGuide, Reply, ToolCall, run_agent
from legaleval_agent.judge_model import Verdict
from legaleval_agent.tool import JudgeTool, ToolError


class ScriptedModel:
    def __init__(self, replies):
        self.queue = list(replies)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        return self.queue.pop(0)


GUIDE = LengthGuide(typical=176, maximum=842)


def _call(text):
    return Reply(text="", tool_call=ToolCall("judge", {"extracted_text": text}))


def _tool(scores):
    queue = list(scores)
    return lambda arguments: Verdict(score=queue.pop(0), typical_score_for_block=4.5, reasons=["short"])


def test_stops_when_the_score_reaches_the_threshold():
    chat = ScriptedModel([_call("first"), _call("second")])
    run = run_agent(
        chat, _tool([2.5, 4.3]), "decision", BLOCKS[0], threshold=4.0, max_judge_calls=3, length=GUIDE
    )
    assert run.attempts == [("first", 2.5), ("second", 4.3)]
    assert run.best == ("second", 4.3)
    assert "reached the threshold" in run.stopped_because
    assert chat.seen[1][-1]["role"] == "tool"
    assert '"score": 2.5' in chat.seen[1][-1]["content"]


def test_gives_up_after_the_maximum_number_of_judge_calls():
    chat = ScriptedModel([_call("a"), _call("b"), _call("c")])
    run = run_agent(
        chat, _tool([2.0, 2.5, 3.0]), "decision", BLOCKS[0], threshold=4.0, max_judge_calls=3, length=GUIDE
    )
    assert len(run.attempts) == 3
    assert run.best == ("c", 3.0)
    assert run.stopped_because == "used all 3 judge calls"


def test_tool_errors_count_toward_the_limit():
    chat = ScriptedModel([_call("a"), _call("b")])
    run = run_agent(
        chat, lambda arguments: ToolError("bad"), "decision", BLOCKS[0], 4.0, max_judge_calls=2, length=GUIDE
    )
    assert run.attempts == []
    assert run.best is None
    assert run.stopped_because == "used all 2 judge calls"


def test_nudges_once_then_stops_if_the_model_never_calls_the_tool():
    chat = ScriptedModel([Reply("here is my answer", None), Reply("still no tool", None)])
    run = run_agent(chat, _tool([]), "decision", BLOCKS[0], threshold=4.0, max_judge_calls=3, length=GUIDE)
    assert run.attempts == []
    assert run.stopped_because == "the model stopped calling the judge tool"
    assert chat.seen[1][-1] == {"role": "user", "content": "Call the judge tool with your extraction."}


def test_judge_tool_scores_text_and_rejects_empty_text(judge, lemmatizer):
    tool = JudgeTool(judge, "решение суда", BLOCKS[0], lemmatizer)
    assert isinstance(tool({"extracted_text": "  "}), ToolError)
    assert isinstance(tool({}), ToolError)
    assert isinstance(tool({"extracted_text": "суд решил"}), Verdict)


def test_judge_tool_refuses_an_unknown_block_at_construction(judge, lemmatizer):
    with pytest.raises(ValueError, match="unknown block"):
        JudgeTool(judge, "решение суда", "nope", lemmatizer)


def test_repeated_extraction_is_not_judged_and_stops_on_the_second_repeat():
    chat = ScriptedModel([_call("same"), _call("same"), _call("same")])
    calls = []

    def tool(arguments):
        calls.append(arguments)
        return Verdict(score=2.0, typical_score_for_block=3.3, reasons=[])

    run = run_agent(chat, tool, "decision", BLOCKS[0], threshold=3.3, max_judge_calls=3, length=GUIDE)
    assert len(calls) == 1
    assert run.attempts == [("same", 2.0)]
    assert "identical to one you already submitted" in chat.seen[2][-1]["content"]
    assert run.stopped_because == "the model resubmitted an earlier extraction 2 times"


def test_judge_tool_refuses_extractions_longer_than_anything_seen_in_training(judge, lemmatizer):
    tool = JudgeTool(judge, "решение суда", BLOCKS[0], lemmatizer)
    too_long = "суд " * (judge.max_chars[BLOCKS[0]] // 4 + 10)
    result = tool({"extracted_text": too_long})
    assert isinstance(result, ToolError)
    assert "cannot score" in result.error


def test_a_cut_off_tool_call_stops_with_an_explicit_reason():
    chat = ScriptedModel(
        [Reply("<tool_call>\n<function=judge>\n<parameter=extracted_text>\nдлинный", None, True)]
    )
    run = run_agent(chat, _tool([]), "decision", BLOCKS[0], threshold=3.3, max_judge_calls=3, length=GUIDE)
    assert run.attempts == []
    assert run.stopped_because == "the reply was cut off at the token limit before the tool call closed"


def test_the_task_states_the_typical_and_maximum_length():
    chat = ScriptedModel([_call("a")])
    run_agent(chat, _tool([5.0]), "decision", BLOCKS[0], threshold=3.3, max_judge_calls=3, length=GUIDE)
    task = chat.seen[0][1]["content"]
    assert "about 176 characters" in task
    assert "must not exceed 842 characters" in task
