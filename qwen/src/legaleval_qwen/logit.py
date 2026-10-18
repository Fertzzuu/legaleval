import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from legaleval_qwen.judge import Case
from legaleval_qwen.prompt import MAX_GRADE, MIN_GRADE, PROMPT_VERSION, messages

GRADES = tuple(range(MIN_GRADE, MAX_GRADE + 1))
PREFILL_STEP = 2048


@dataclass(frozen=True)
class LogitJudgment:
    case_id: str
    model: str
    prompt_version: str
    probabilities: dict[int, list[float]]
    grade_mass: dict[int, float]
    prompt_tokens: int
    seconds: float

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "LogitJudgment":
        probabilities = {int(k): v for k, v in record["probabilities"].items()}
        mass = {int(k): v for k, v in record["grade_mass"].items()}
        return cls(**{**record, "probabilities": probabilities, "grade_mass": mass})

    def expected_grades(self) -> dict[int, float]:
        return {
            n: float(sum(g * p for g, p in zip(GRADES, probs, strict=True)))
            for n, probs in self.probabilities.items()
        }


def answer_opening(first_block: int) -> str:
    return '{\n  "' + f'{first_block}": '


def answer_step(grade: int, next_block: int) -> str:
    return f'{grade},\n  "{next_block}": '


Scorer = Callable[[list[dict[str, str]], list[int]], tuple[dict[int, list[float]], dict[int, float], int]]


def mlx_logit_scorer(model_id: str) -> Scorer:
    model, tokenizer = load(model_id)
    grade_ids = []
    for grade in GRADES:
        ids = tokenizer.encode(str(grade), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"grade {grade} is not a single token for {model_id}: {ids}")
        grade_ids.append(ids[0])

    def feed(cache: list[Any], ids: list[int]) -> mx.array:
        if not ids:
            raise ValueError("nothing to feed to the model")
        starts = range(0, len(ids), PREFILL_STEP)
        for start in starts[:-1]:
            mx.eval(model(mx.array(ids[start : start + PREFILL_STEP])[None], cache=cache))
        logits = model(mx.array(ids[starts[-1] :])[None], cache=cache)
        mx.eval(logits)
        return logits[0, -1]

    def run(
        chat: list[dict[str, str]], blocks: list[int]
    ) -> tuple[dict[int, list[float]], dict[int, float], int]:
        prompt = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        text = prompt + answer_opening(blocks[0])
        fed = tokenizer.encode(text, add_special_tokens=False)
        prompt_tokens = len(fed)
        cache = make_prompt_cache(model)
        logits = feed(cache, fed)
        probabilities, mass = {}, {}
        for i, block in enumerate(blocks):
            candidates = np.array(mx.softmax(logits.astype(mx.float32))[mx.array(grade_ids)])
            mass[block] = float(candidates.sum())
            probabilities[block] = (candidates / candidates.sum()).tolist()
            if i + 1 < len(blocks):
                step = answer_step(GRADES[int(candidates.argmax())], blocks[i + 1])
                ids = tokenizer.encode(step, add_special_tokens=False)
                text += step
                fed += ids
                logits = feed(cache, ids)
        if tokenizer.encode(text, add_special_tokens=False) != fed:
            raise ValueError("the forced answer tokenizes differently from what was fed to the model")
        return probabilities, mass, prompt_tokens

    return run


def judge_case_logits(case: Case, model_id: str, scorer: Scorer) -> LogitJudgment:
    started = time.perf_counter()
    probabilities, mass, prompt_tokens = scorer(
        messages(case.decision, case.blocks), [b.number for b in case.blocks]
    )
    return LogitJudgment(
        case_id=case.case_id,
        model=model_id,
        prompt_version=PROMPT_VERSION,
        probabilities=probabilities,
        grade_mass=mass,
        prompt_tokens=prompt_tokens,
        seconds=time.perf_counter() - started,
    )
