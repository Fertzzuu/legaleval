import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from legaleval_gbm.cv import ModelFactory, block_prior, fit_predict, gbm, gbm_ridge_blend, run_cv
from legaleval_gbm.data import (
    BLOCK,
    BLOCKS,
    CASE_ID,
    DEFAULT_DATA_DIR,
    ROW_ID,
    TARGET,
    load_sample_submission,
    load_split,
)
from legaleval_gbm.features import FEATURE_SETS

from legaleval_qwen.calibrate import GRADE, isotonic_grade, linear_grade_by_block, raw_grade
from legaleval_qwen.judge import (
    Judgment,
    JudgmentStore,
    cases,
    grades_frame,
    judge_case,
    mlx_generator,
    text_grades,
)
from legaleval_qwen.logit import LogitJudgment, judge_case_logits, mlx_logit_scorer
from legaleval_qwen.prompt import PROMPT_VERSION
from legaleval_qwen.reporting import check_submission, summarize

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_GBM_OUTPUTS = PROJECT_DIR.parent / "gbm" / "outputs"

GBM_COLUMNS = list(FEATURE_SETS["all"])
METHODS: dict[str, tuple[ModelFactory, list[str]]] = {
    "raw_grade": (raw_grade, [GRADE]),
    "isotonic_grade": (isotonic_grade, [GRADE]),
    "linear_grade_by_block": (linear_grade_by_block, [GRADE, BLOCK]),
    "gbm_all": (gbm, GBM_COLUMNS),
    "gbm_all+llm": (gbm, [*GBM_COLUMNS, GRADE]),
    "blend_all": (gbm_ridge_blend, GBM_COLUMNS),
    "blend_all+llm": (gbm_ridge_blend, [*GBM_COLUMNS, GRADE]),
}


def _file_name(model: str, split: str) -> str:
    return f"{model.replace('/', '__')}_{PROMPT_VERSION}_{split}.jsonl"


def _store(out_dir: Path, model: str, split: str) -> JudgmentStore[Judgment]:
    return JudgmentStore(out_dir / "judgments" / _file_name(model, split), Judgment.from_record)


def _logit_store(out_dir: Path, model: str, split: str) -> JudgmentStore[LogitJudgment]:
    return JudgmentStore(out_dir / "logits" / _file_name(model, split), LogitJudgment.from_record)


def _grades(args: argparse.Namespace, split: str) -> dict[str, dict[int, float]]:
    if args.source == "logit":
        return {
            c: j.expected_grades() for c, j in _logit_store(args.out_dir, args.model, split).load().items()
        }
    return text_grades(_store(args.out_dir, args.model, split).load())


def _judge(args: argparse.Namespace) -> None:
    df = load_split(args.data_dir, args.split)
    store = _store(args.out_dir, args.model, args.split)
    done = {c for c, j in store.load().items() if j.grades is not None}
    todo = [c for c in cases(df) if c.case_id not in done][: args.limit]
    print(f"{len(done)} cases already judged, {len(todo)} to go -> {store.path}")
    if not todo:
        return
    generator = mlx_generator(args.model, args.max_tokens)
    failures = 0
    for i, case in enumerate(todo, start=1):
        judgment = judge_case(case, args.model, generator)
        store.append(judgment)
        failures += judgment.grades is None
        outcome = judgment.grades if judgment.grades is not None else f"FAILED: {judgment.error}"
        print(
            f"[{i}/{len(todo)}] {case.case_id} {judgment.prompt_tokens} tokens "
            f"{judgment.seconds:.1f}s {outcome}",
            flush=True,
        )
    if failures:
        print(f"{failures} cases failed to parse; rerun `judge` to retry them, or inspect {store.path}")


def _logit_judge(args: argparse.Namespace) -> None:
    df = load_split(args.data_dir, args.split)
    store = _logit_store(args.out_dir, args.model, args.split)
    done = set(store.load())
    todo = [c for c in cases(df) if c.case_id not in done][: args.limit]
    print(f"{len(done)} cases already scored, {len(todo)} to go -> {store.path}")
    if not todo:
        return
    scorer = mlx_logit_scorer(args.model)
    for i, case in enumerate(todo, start=1):
        judgment = judge_case_logits(case, args.model, scorer)
        store.append(judgment)
        expected = {n: round(g, 2) for n, g in judgment.expected_grades().items()}
        low_mass = min(judgment.grade_mass.values())
        print(
            f"[{i}/{len(todo)}] {case.case_id} {judgment.prompt_tokens} tokens "
            f"{judgment.seconds:.1f}s {expected} min grade mass {low_mass:.2f}",
            flush=True,
        )


def _features(args: argparse.Namespace, split: str, df: pd.DataFrame) -> pd.DataFrame:
    path = args.gbm_outputs / f"features_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run legaleval-gbm` in ../gbm first")
    features = pd.read_csv(path).drop(columns=[TARGET], errors="ignore")
    if list(features[ROW_ID]) != list(df[ROW_ID]):
        raise ValueError(f"{path} rows do not match {split}.csv")
    features[BLOCK] = pd.Categorical(features[BLOCK], categories=BLOCKS)
    grades = grades_frame(df, _grades(args, split))
    return features.merge(grades, on=ROW_ID, how="left", validate="one_to_one")


def _evaluate(args: argparse.Namespace) -> None:
    train = load_split(args.data_dir, "train")
    X = _features(args, "train", train)
    y = train[TARGET].to_numpy()
    seeds = list(range(args.seeds))
    print(f"expert score (rounded) by LLM grade (rounded), source={args.source}:")
    print(pd.crosstab(X[GRADE].round(), train[TARGET].round()).to_string())
    results = [run_cv("block_prior", block_prior, X, y, X[CASE_ID], seeds, args.folds)]
    results += [
        run_cv(name, factory, X[columns], y, X[CASE_ID], seeds, args.folds)
        for name, (factory, columns) in METHODS.items()
    ]
    summary = pd.DataFrame([row for r in results for row in summarize(r)])
    print(f"\n{args.folds}-fold GroupKFold by case x {len(seeds)} seeds")
    print(summary.round(4).to_string(index=False))


def _submit(args: argparse.Namespace) -> None:
    train, test = load_split(args.data_dir, "train"), load_split(args.data_dir, "test")
    factory, columns = METHODS[args.method]
    X_train, X_test = _features(args, "train", train)[columns], _features(args, "test", test)[columns]
    predictions = fit_predict(factory, X_train, train[TARGET].to_numpy(), X_test, list(range(args.seeds)))
    submission = pd.DataFrame({ROW_ID: test[ROW_ID], "predicted_score": predictions})
    check_submission(submission, load_sample_submission(args.data_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{args.source}_{args.method.replace('+', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    path = args.out_dir / f"submission_qwen_{stamp}.csv"
    submission.to_csv(path, index=False)
    print(f"submission written to {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen zero-shot judge")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", required=True, help="MLX model repo, e.g. mlx-community/Qwen3.8-27B-4bit")
    commands = parser.add_subparsers(dest="command", required=True)

    judge = commands.add_parser("judge", help="grade every case of a split (resumable)")
    judge.add_argument("--split", choices=["train", "test"], required=True)
    judge.add_argument("--limit", type=int, default=None)
    judge.add_argument("--max-tokens", type=int, default=64)
    judge.set_defaults(run=_judge)

    logit = commands.add_parser(
        "logit-judge", help="score every case from grade-token probabilities (resumable)"
    )
    logit.add_argument("--split", choices=["train", "test"], required=True)
    logit.add_argument("--limit", type=int, default=None)
    logit.set_defaults(run=_logit_judge)

    for name, run in (("evaluate", _evaluate), ("submit", _submit)):
        command = commands.add_parser(name)
        command.add_argument("--source", choices=["text", "logit"], default="text")
        command.add_argument("--gbm-outputs", type=Path, default=DEFAULT_GBM_OUTPUTS)
        command.add_argument("--seeds", type=int, default=5)
        command.add_argument("--folds", type=int, default=5)
        command.set_defaults(run=run)
    commands.choices["submit"].add_argument("--method", choices=sorted(METHODS), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.run(args)
