import json
import re
from dataclasses import dataclass

from legaleval_gbm.data import BLOCKS

PROMPT_VERSION = "v1"
MIN_GRADE = 1
MAX_GRADE = 5

BLOCK_GUIDE: dict[str, str] = dict(
    zip(
        BLOCKS,
        (
            "what the plaintiff asks the court to award or order",
            "the plaintiff's arguments and the grounds for the claims",
            "the defendant's position and objections, or that the defendant gave none",
            "how the court assessed the evidence the parties submitted",
            "the chain of the judge's reasoning and intermediate conclusions",
            "the laws and articles the court applied",
            "the court's final ruling (the operative part)",
        ),
        strict=True,
    )
)

SYSTEM_PROMPT = """You are an experienced Russian lawyer reviewing automatic extractions from court decisions.
For each block you get the block name, what the block should contain, and the extracted text.
Grade how well the extracted text captures that block of THIS decision:

5 - complete and accurate: all essential content of the block, nothing invented, nothing from other blocks
4 - accurate with minor omissions or minor imprecision
3 - partly correct: important omissions, or noticeable content that belongs to another block
2 - mostly wrong, very incomplete, or largely about something else
1 - fails the task: wrong content, invented facts, or nothing usable

Judge only against the decision text. Answer with one JSON object mapping each block number \
to an integer grade, for example {"1": 5, "2": 3}. Output nothing else."""


@dataclass(frozen=True)
class Block:
    number: int
    name: str
    extracted: str


class JudgeParseError(ValueError):
    pass


def user_message(decision: str, blocks: list[Block]) -> str:
    parts = ["DECISION:", decision.strip(), "", "BLOCKS:"]
    for b in blocks:
        parts += [
            f"[{b.number}] {b.name} - {BLOCK_GUIDE[b.name]}",
            f"Extracted text: {b.extracted.strip()}",
            "",
        ]
    numbers = ", ".join(f'"{b.number}"' for b in blocks)
    parts.append(f"Return JSON with keys {numbers}.")
    return "\n".join(parts)


def messages(decision: str, blocks: list[Block]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(decision, blocks)},
    ]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def parse_grades(response: str, blocks: list[Block]) -> dict[int, int]:
    candidates = _OBJECT_RE.findall(_THINK_RE.sub("", response))
    if not candidates:
        raise JudgeParseError("no JSON object in response")
    try:
        raw = json.loads(candidates[-1])
    except json.JSONDecodeError as error:
        raise JudgeParseError(f"invalid JSON: {error}") from error
    expected = {str(b.number) for b in blocks}
    if set(raw) != expected:
        raise JudgeParseError(f"keys {sorted(raw)} != expected {sorted(expected)}")
    grades = {int(k): v for k, v in raw.items()}
    bad = {k: v for k, v in grades.items() if type(v) is not int or not MIN_GRADE <= v <= MAX_GRADE}
    if bad:
        raise JudgeParseError(f"grades outside {MIN_GRADE}-{MAX_GRADE}: {bad}")
    return grades
