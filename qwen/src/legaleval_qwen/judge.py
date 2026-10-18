import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

import pandas as pd
from legaleval_gbm.data import BLOCK, BLOCKS, CASE_ID, EXTRACTED, ROW_ID, SOURCE
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from legaleval_qwen.calibrate import GRADE
from legaleval_qwen.prompt import PROMPT_VERSION, Block, JudgeParseError, messages, parse_grades


@dataclass(frozen=True)
class Case:
    case_id: str
    decision: str
    blocks: list[Block]
    row_ids: dict[int, str]


@dataclass(frozen=True)
class Judgment:
    case_id: str
    model: str
    prompt_version: str
    response: str
    grades: dict[int, int] | None
    error: str | None
    prompt_tokens: int
    seconds: float

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Judgment":
        grades = record["grades"]
        parsed = {int(k): v for k, v in grades.items()} if grades is not None else None
        return cls(**{**record, "grades": parsed})


class CaseRecord(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def case_id(self) -> str: ...


def cases(df: pd.DataFrame) -> Iterator[Case]:
    order = {name: i for i, name in enumerate(BLOCKS)}
    for case_id, rows in df.groupby(CASE_ID, sort=False):
        rows = rows.sort_values(BLOCK, key=lambda s: s.map(order))
        numbered = list(enumerate(rows.itertuples(index=False), start=1))
        yield Case(
            case_id=str(case_id),
            decision=rows[SOURCE].iloc[0],
            blocks=[Block(n, getattr(r, BLOCK), getattr(r, EXTRACTED)) for n, r in numbered],
            row_ids={n: getattr(r, ROW_ID) for n, r in numbered},
        )


class JudgmentStore[Record: CaseRecord]:
    def __init__(self, path: Path, decode: Callable[[dict[str, Any]], Record]) -> None:
        self.path = path
        self.decode = decode

    def load(self) -> dict[str, Record]:
        if not self.path.exists():
            return {}
        latest: dict[str, Record] = {}
        for line in self.path.read_text().splitlines():
            record = self.decode(json.loads(line))
            latest[record.case_id] = record
        return latest

    def append(self, record: Record) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


Generator = Callable[[list[dict[str, str]]], tuple[str, int]]


def mlx_generator(model_id: str, max_tokens: int) -> Generator:
    model, tokenizer = load(model_id)
    sampler = make_sampler(temp=0.0)

    def run(chat: list[dict[str, str]]) -> tuple[str, int]:
        prompt = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        response = generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler)
        return response, len(tokenizer.encode(prompt))

    return run


def judge_case(case: Case, model_id: str, generator: Generator) -> Judgment:
    started = time.perf_counter()
    response, prompt_tokens = generator(messages(case.decision, case.blocks))
    try:
        grades, error = parse_grades(response, case.blocks), None
    except JudgeParseError as failure:
        grades, error = None, str(failure)
    return Judgment(
        case_id=case.case_id,
        model=model_id,
        prompt_version=PROMPT_VERSION,
        response=response,
        grades=grades,
        error=error,
        prompt_tokens=prompt_tokens,
        seconds=time.perf_counter() - started,
    )


def text_grades(judgments: dict[str, Judgment]) -> dict[str, dict[int, float]]:
    return {
        c: {n: float(g) for n, g in j.grades.items()} for c, j in judgments.items() if j.grades is not None
    }


def grades_frame(df: pd.DataFrame, grades_by_case: dict[str, dict[int, float]]) -> pd.DataFrame:
    missing = sorted(c for c in df[CASE_ID].unique() if c not in grades_by_case)
    if missing:
        raise ValueError(f"{len(missing)} cases have no valid judgment, e.g. {missing[:5]}; rerun the judge")
    rows = [
        {ROW_ID: case.row_ids[n], GRADE: grade}
        for case in cases(df)
        for n, grade in grades_by_case[case.case_id].items()
    ]
    return pd.DataFrame(rows)
