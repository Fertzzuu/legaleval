# Agent with a GBM judge tool

A local Qwen model (MLX) extracts one block from a court decision and checks its own work with a
`judge` tool. The judge is the GBM-ridge blend on lexical features (CV 0.762), trained on the competition's
706 expert-scored rows. If the score is below the block's typical expert score, the agent revises and
calls the judge again.

This is a demonstration of a quality gate, not a leaderboard submission: no expert has scored the agent's
own extractions, so a higher judge score is not proof of a better extraction. Read the transcripts.

## Run

Depends on `../gbm` (editable path dependency) for features and models; data is read from `../data/`.

```bash
uv sync
uv run legaleval-agent train-judge
uv run legaleval-agent run --case SR00354 --block "Оценка судом представленных сторонами доказательств"
uv run legaleval-agent compare --cases 5 --block "Оценка судом представленных сторонами доказательств"
uv run pytest
```

`run` and `compare` take `--model` (default `mlx-community/Qwen3.5-4B-MLX-4bit`), `--threshold`
(default: the block's typical expert score), `--max-calls` (default 3) and `--max-tokens` (default 2048).
Transcripts are written to `outputs/agent_runs/`.

## How the loop works

1. The agent gets the decision, the block to extract, and the block's typical and maximum length.
2. It calls `judge(extracted_text)`. The harness binds the decision and the block; the model only sends
   its text.
3. The judge returns a 1-5 score, the block's typical expert score, and reasons when the extraction is
   unusual (shorter or longer than 90% of training extractions, many words not in the decision).
4. The run stops when the score reaches the threshold, after `--max-calls` judge calls, when the agent
   resubmits identical text twice, when it stops calling the tool, or when a reply is cut off at the
   token limit.

## Safeguards

| Safeguard | Why |
|-----------|-----|
| Frozen rescale | A single extraction gets the same score alone as in a batch |
| Out-of-range guard | The judge refuses extractions longer than any training extraction for that block, because length is its strongest feature and it saturates beyond the training range |
| Repeat detection | Identical resubmissions are not judged; with greedy decoding the agent otherwise repeats itself |
| Bound block | The block name is not a tool argument; a 4B model misspelled the long Russian name |

| Module | Role |
|--------|------|
| `judge_model.py` | Train, save and load the judge; frozen rescale; reasons; out-of-range guard |
| `tool.py` | The `judge` tool schema and the callable bound to one decision and block |
| `agent.py` | MLX chat model with tool calling, and the agent loop |
| `cli.py` | `train-judge`, `run`, `compare` |
