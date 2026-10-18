import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from legaleval_agent.judge_model import Verdict
from legaleval_agent.tool import JUDGE_TOOL, ToolError

SYSTEM_PROMPT = """You extract one block of information from a Russian court decision.
Write the extraction in Russian, using the decision's own wording where you can, and cover everything
the block should contain, without content that belongs to other blocks.
After writing an extraction, always check it by calling the judge tool. If the score is below {threshold},
read the reasons, improve the extraction and call the judge tool again."""

USER_PROMPT = """DECISION:
{decision}

Extract the block "{block_name}" from this decision and check it with the judge tool.
A typical extraction for this block is about {typical} characters long;
it must not exceed {maximum} characters.
Keep only the essential content of the block."""

NUDGE = "Call the judge tool with your extraction."
REPEAT = (
    "This extraction is identical to one you already submitted. Revise it before calling the judge again."
)
MAX_REPEATS = 2


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LengthGuide:
    typical: int
    maximum: int


@dataclass(frozen=True)
class Reply:
    text: str
    tool_call: ToolCall | None
    cut_off: bool = False


ChatModel = Callable[[list[dict[str, object]], list[dict[str, object]]], Reply]
Tool = Callable[[dict[str, object]], Verdict | ToolError]


@dataclass
class AgentRun:
    attempts: list[tuple[str, float]] = field(default_factory=list)
    transcript: list[dict[str, object]] = field(default_factory=list)
    stopped_because: str = ""

    @property
    def best(self) -> tuple[str, float] | None:
        return max(self.attempts, key=lambda a: a[1]) if self.attempts else None


def mlx_chat_model(model_id: str, max_tokens: int) -> ChatModel:
    model, tokenizer = load(model_id)
    sampler = make_sampler(temp=0.0)
    start, end = tokenizer.tool_call_start, tokenizer.tool_call_end

    def chat(messages: list[dict[str, object]], tools: list[dict[str, object]]) -> Reply:
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        text = generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler)
        if start in text and end not in text:
            return Reply(text=text, tool_call=None, cut_off=True)
        if start not in text:
            return Reply(text=text, tool_call=None)
        body = text.split(start, 1)[1].split(end, 1)[0]
        parsed = tokenizer.tool_parser(body, tools)
        return Reply(text=text, tool_call=ToolCall(name=parsed["name"], arguments=dict(parsed["arguments"])))

    return chat


def run_agent(
    chat_model: ChatModel,
    tool: Tool,
    decision: str,
    block_name: str,
    threshold: float,
    max_judge_calls: int,
    length: LengthGuide,
) -> AgentRun:
    run = AgentRun()
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(threshold=threshold)},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                decision=decision, block_name=block_name, typical=length.typical, maximum=length.maximum
            ),
        },
    ]
    nudged = False
    calls = 0
    repeats = 0
    submitted: set[str] = set()
    while calls < max_judge_calls:
        reply = chat_model(messages, [JUDGE_TOOL])
        if reply.cut_off:
            run.transcript.append({"role": "assistant", "content": reply.text})
            run.stopped_because = "the reply was cut off at the token limit before the tool call closed"
            return run
        if reply.tool_call is None or reply.tool_call.name != "judge":
            run.transcript.append({"role": "assistant", "content": reply.text})
            if nudged:
                run.stopped_because = "the model stopped calling the judge tool"
                return run
            messages += [{"role": "assistant", "content": reply.text}, {"role": "user", "content": NUDGE}]
            nudged = True
            continue
        nudged = False
        text = str(reply.tool_call.arguments.get("extracted_text", "")).strip()
        if text in submitted:
            repeats += 1
            result: Verdict | ToolError = ToolError(REPEAT)
        else:
            submitted.add(text)
            calls += 1
            result = tool(reply.tool_call.arguments)
        payload = asdict(result)
        call = {"type": "function", "function": {"name": "judge", "arguments": reply.tool_call.arguments}}
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "name": "judge", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        run.transcript += [
            {"role": "assistant", "tool_call": reply.tool_call.arguments},
            {"role": "tool", **payload},
        ]
        if repeats >= MAX_REPEATS:
            run.stopped_because = f"the model resubmitted an earlier extraction {MAX_REPEATS} times"
            return run
        if isinstance(result, ToolError):
            continue
        run.attempts.append((str(reply.tool_call.arguments["extracted_text"]), result.score))
        if result.score >= threshold:
            run.stopped_because = f"score {result.score} reached the threshold {threshold}"
            return run
    run.stopped_because = f"used all {max_judge_calls} judge calls"
    return run
