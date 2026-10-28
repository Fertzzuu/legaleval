# LegalEval Extract: LLM-only solution

One script that predicts the expert score (1-5) for each extracted block of a court decision, using only
a language model as the judge: no gradient boosting and no other trained model.

**Result: a documented negative result.** The judge scores below simply predicting each block's average
expert score (see [Score](#score)). Use `../simple/` or `../gbm/` for a real submission.

## How it works

1. **Embeddings** (E5, `intfloat/multilingual-e5-large`): every decision is cut into 80-word passages,
   and the passages, the extractions and the block names are turned into vectors. Similar texts get
   similar vectors, so a dot product finds the closest ones.
2. **Prompt** (one per block): the block name, the decision passage closest to the extraction and the one
   closest to the block name (in document order, with their position), the 2 most similar extractions of
   the same block from *other* training cases with their expert scores, and the extraction to grade.
3. **Judge** (`Qwen3.5-4B`, local, MLX): instead of letting the model write an answer, the script reads its
   probabilities for answering "1", "2", "3", "4" or "5" and takes the expected grade
   (for example 0.1 x 3 + 0.5 x 4 + 0.4 x 5 = 4.3). This gives finer scores than a written digit.
4. **Rescaling**: grades are stretched to the mean and spread of the expert scores, because the metric's
   CCC part punishes predictions that are bunched around the average.

The training score is a fair estimate even though no cross-validation is needed: nothing is fitted, and
the examples for a training row always come from other cases. Only the two rescaling numbers (the
experts' mean and spread) come from the training labels.

## Run

Needs a Mac with Apple Silicon (MLX), Python 3.13 (the version it was tested with), and about 6 GB of disk
for the two models (Qwen 3 GB, E5 2.2 GB; downloaded on the first run unless already cached).

Put `train.csv`, `test.csv` and `sample_submission.csv` in `../data/`, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py --cases 3
python solution.py --cases 101
python solution.py
```

`--cases N` grades only the first N training cases, prints their score and skips the test set:
`--cases 3` is a quick check that everything works, `--cases 101` scores the whole training set
(about 10 minutes on an M4 Max). Without `--cases` the script also grades the test set and writes
`outputs/submission.csv`; that has not been run (estimated at about 50 more minutes). The first
run also spends about 11 minutes embedding the decisions. The first prompt the model saw is written to
`outputs/example_prompt.txt`.

The laptop gets hot during long runs: the GPU is fully busy the whole time.

If the Qwen weights came from the shipped zip, unzip them into `~/.cache/huggingface/hub` and run with
`HF_HUB_OFFLINE=1` so nothing is downloaded.

## Saved work

The run can be stopped and started again: it continues where it stopped. `--cases` with a growing number
(for example 30, then 60, then 101) grades the training set in smaller bites.

| File in `outputs/` | Holds | Delete it when |
|--------------------|-------|----------------|
| `passage_vectors_80_words/` | one file of passage vectors per decision | you change the embedding model |
| `train_extraction_vectors.npy`, `test_extraction_vectors.npy` | extraction vectors | you change the embedding model |
| `grades_train.csv`, `grades_test.csv` | the judge's grade for every row graded so far | you change the prompt, the examples or the LLM |

These files are derived from the competition data, so keep them on your own machine.

## Score

Scored with the competition formula (0.55 x CCC + 0.30 x Pearson + 0.15 x normalized MAE; the MAE part
is assumed to be `1 - MAE / 4`) on all 706 training rows. The range is a 95% bootstrap over cases.

| Version | Score | Range |
|---------|-------|-------|
| Trimmed grounded: 2 passages, 2 examples (**this script**) | 0.362 | 0.30-0.42 |
| Full grounded: 6 passages, 4 examples | 0.387 | 0.32-0.44 |
| Few-shot only: no passages, 4 examples | 0.276 | 0.22-0.33 |
| Each block's average expert score (no LLM) | 0.546 | 0.46-0.63 |
| Average expert score of the 4 most similar examples (no LLM) | 0.679 | 0.62-0.73 |
| GBM-ridge blend (`../gbm`) | 0.762 | |

What the numbers say:

- **Showing decision passages helps**: both grounded versions beat few-shot only by about 0.09-0.11.
- **2 or 6 passages makes no measurable difference**: full minus trimmed is +0.025 (range -0.020 to
  +0.069), so the script keeps the trimmed prompt, which runs twice as fast.
- **The LLM makes its own examples worse**: every version scores 0.29-0.40 below the plain average of the
  examples it was shown. The 4B model adds noise to the signal in its prompt instead of improving it.
- **No version reaches the block average**, which needs no model at all (0.546, computed on the same
  rows; about 0.53 when the averages come from other cases).

The first 87 rows looked much better (0.58 for the full grounded prompt) and were misleading: they come
from only 13 cases. Judge a prompt change on the whole training set.
