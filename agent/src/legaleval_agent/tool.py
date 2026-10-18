from dataclasses import dataclass

from legaleval_gbm.data import BLOCKS
from legaleval_gbm.text import Lemmatizer

from legaleval_agent.judge_model import JudgeModel, OutOfRange, Verdict

JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "judge",
        "description": (
            "Score how well your extracted text captures the block you are extracting from the current "
            "court decision, on the 1-5 scale experts use (5 = complete and accurate). Returns the score, "
            "the typical expert score for that block, and reasons when the extraction looks unusual."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "extracted_text": {
                    "type": "string",
                    "description": "Your extracted text for the block, in Russian.",
                },
            },
            "required": ["extracted_text"],
        },
    },
}


@dataclass(frozen=True)
class ToolError:
    error: str


class JudgeTool:
    name = "judge"

    def __init__(self, judge: JudgeModel, decision: str, block_name: str, lemmatizer: Lemmatizer) -> None:
        if block_name not in BLOCKS:
            raise ValueError(f"unknown block {block_name!r}")
        self.judge = judge
        self.decision = decision
        self.block_name = block_name
        self.lemmatizer = lemmatizer

    def __call__(self, arguments: dict[str, object]) -> Verdict | ToolError:
        extracted = arguments.get("extracted_text")
        if not isinstance(extracted, str) or not extracted.strip():
            return ToolError("extracted_text must be a non-empty string")
        verdict = self.judge.judge(self.decision, self.block_name, extracted, self.lemmatizer)
        if isinstance(verdict, OutOfRange):
            return ToolError(verdict.reason)
        return verdict
