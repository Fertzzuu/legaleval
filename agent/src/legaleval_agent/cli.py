import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from legaleval_gbm.data import BLOCK, BLOCKS, CASE_ID, DEFAULT_DATA_DIR, EXTRACTED, SOURCE, TARGET, load_split
from legaleval_gbm.text import Lemmatizer

from legaleval_agent.agent import AgentRun, ChatModel, LengthGuide, mlx_chat_model, run_agent
from legaleval_agent.judge_model import JudgeModel, train_judge
from legaleval_agent.tool import JudgeTool, ToolError

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"


def _judge_path(args: argparse.Namespace) -> Path:
    return args.out_dir / "judge.joblib"


def _train_judge(args: argparse.Namespace) -> None:
    train = load_split(args.data_dir, "train")
    judge = train_judge(train, Lemmatizer(), list(range(args.seeds)), args.folds)
    judge.save(_judge_path(args))
    totals = np.array([s.total for s in judge.cv_scores])
    print(
        f"judge CV ({args.folds}-fold grouped x {args.seeds} seeds): "
        f"{totals.mean():.4f} +/- {totals.std():.4f}"
    )
    print(
        f"frozen rescale: prediction mean {judge.prediction_mean:.3f}, std {judge.prediction_std:.3f} "
        f"-> expert mean {judge.target_mean:.3f}, std {judge.target_std:.3f}"
    )
    print(f"saved {_judge_path(args)}")


def _row(train: pd.DataFrame, case: str, block: str) -> pd.Series:
    match = train[(train[CASE_ID] == case) & (train[BLOCK] == block)]
    if match.empty:
        raise ValueError(f"no training row for case {case!r} and block {block!r}")
    return match.iloc[0]


def _agent(
    args: argparse.Namespace, judge: JudgeModel, lemmatizer: Lemmatizer, chat: ChatModel, row: pd.Series
) -> AgentRun:
    tool = JudgeTool(judge, row[SOURCE], row[BLOCK], lemmatizer)
    threshold = args.threshold if args.threshold is not None else round(judge.block_scores[row[BLOCK]], 2)
    length = LengthGuide(
        typical=int(judge.ranges[row[BLOCK]]["ext_chars"].typical), maximum=judge.max_chars[row[BLOCK]]
    )
    return run_agent(chat, tool, row[SOURCE], row[BLOCK], threshold, args.max_calls, length)


def _save_run(args: argparse.Namespace, row: pd.Series, run: AgentRun) -> Path:
    path = args.out_dir / "agent_runs" / f"{row[CASE_ID]}_{BLOCKS.index(row[BLOCK]) + 1}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "case": row[CASE_ID],
        "block": row[BLOCK],
        "stopped_because": run.stopped_because,
        "attempts": run.attempts,
        "transcript": run.transcript,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return path


def _run(args: argparse.Namespace) -> None:
    train = load_split(args.data_dir, "train")
    row = _row(train, args.case, args.block)
    judge, lemmatizer = JudgeModel.load(_judge_path(args)), Lemmatizer()
    run = _agent(args, judge, lemmatizer, mlx_chat_model(args.model, args.max_tokens), row)
    for step in run.transcript:
        if "tool_call" in step:
            print(f"\n[agent -> judge]\n{step['tool_call'].get('extracted_text')}")
        elif step["role"] == "tool":
            print(f"[judge] {json.dumps({k: v for k, v in step.items() if k != 'role'}, ensure_ascii=False)}")
        else:
            print(f"\n[agent] {step['content']}")
    print(f"\nstopped: {run.stopped_because}")
    print(f"transcript saved to {_save_run(args, row, run)}")


def _compare(args: argparse.Namespace) -> None:
    train = load_split(args.data_dir, "train")
    judge, lemmatizer = JudgeModel.load(_judge_path(args)), Lemmatizer()
    chat = mlx_chat_model(args.model, args.max_tokens)
    cases = list(dict.fromkeys(train[CASE_ID]))[: args.cases]
    rows = []
    for case in cases:
        row = _row(train, case, args.block)
        original = JudgeTool(judge, row[SOURCE], row[BLOCK], lemmatizer)({"extracted_text": row[EXTRACTED]})
        if isinstance(original, ToolError):
            raise ValueError(f"judge rejected the dataset extraction for {case}: {original.error}")
        run = _agent(args, judge, lemmatizer, chat, row)
        best = run.best
        _save_run(args, row, run)
        rows.append(
            {
                "case": case,
                "expert_score": row[TARGET],
                "judge_on_dataset_extraction": original.score,
                "judge_on_agent_best": best[1] if best else np.nan,
                "judge_calls": len([s for s in run.transcript if "tool_call" in s]),
                "stopped_because": run.stopped_because,
            }
        )
        print(
            f"{case}: dataset {original.score:.2f} | agent {rows[-1]['judge_on_agent_best']:.2f}", flush=True
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(
        "\nNote: experts never scored the agent's extractions; a higher judge score is not proof of a better "
        "extraction. Read the transcripts in outputs/agent_runs/."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen agent with a GBM judge tool")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-judge", help="fit and save the judge model")
    train.add_argument("--seeds", type=int, default=5)
    train.add_argument("--folds", type=int, default=5)
    train.set_defaults(run=_train_judge)

    for name, run, help_text in (
        ("run", _run, "run the agent on one training case and block"),
        ("compare", _compare, "run the agent on several training cases for one block"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--block", choices=list(BLOCKS), required=True)
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument(
            "--threshold", type=float, default=None, help="default: the block's typical expert score"
        )
        command.add_argument("--max-calls", type=int, default=3)
        command.add_argument("--max-tokens", type=int, default=2048)
        command.set_defaults(run=run)
    commands.choices["run"].add_argument("--case", required=True)
    commands.choices["compare"].add_argument("--cases", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.run(args)
