# Qwen zero-shot judge

A local Qwen model (MLX) reads each court decision with its 7 extractions and grades every block 1–5
against a rubric. The grades are calibrated to the expert scale, and also stacked into the GBM as a
feature. Scored with the same repeated `GroupKFold` and metric as `../gbm`.

Depends on `../gbm` (editable path dependency) for data loading, the metric, CV and GBM features.
`evaluate` and `submit` read `../gbm/outputs/features_{train,test}.csv`, so run `uv run legaleval-gbm`
in `../gbm` first.

```bash
uv sync
M=mlx-community/Qwen3.5-4B-MLX-4bit
uv run legaleval-qwen --model $M judge --split train --limit 3
uv run legaleval-qwen --model $M judge --split train
uv run legaleval-qwen --model $M evaluate
uv run legaleval-qwen --model $M judge --split test
uv run legaleval-qwen --model $M submit --method blend_all+llm
uv run pytest
```

`judge` is resumable: it appends every response to
`outputs/judgments/<model>_<prompt version>_<split>.jsonl` and skips cases that already have valid
grades. Responses that fail to parse are kept with their error and retried on the next run.

| Module | Role |
|--------|------|
| `prompt.py` | Rubric, block descriptions, chat messages, strict JSON grade parser |
| `judge.py` | Case assembly, MLX generation (greedy, thinking off), judgment store |
| `calibrate.py` | Raw, isotonic and linear-by-block calibrators (fit/predict like sklearn) |
| `reporting.py` | CV summary rows and submission check (copied from `../gbm`, kept separate on purpose) |
| `cli.py` | `judge`, `evaluate`, `submit` |
