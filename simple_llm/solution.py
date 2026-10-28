"""LegalEval Extract: an LLM-only judge for extracted court-decision blocks.

Judge: Qwen3.5-4B (runs locally with MLX) grades one block at a time. Its prompt shows
- the decision passages that an embedding model (E5) found closest to the extraction and to the block name,
- the extractions of the same block from other training cases that E5 found most similar, with their expert scores.
Score: the expected grade from the model's probabilities for answering "1" to "5", stretched to the experts'
mean and spread. No gradient boosting and no other trained model.
"""

import argparse
import csv
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from fastembed import TextEmbedding
from mlx_lm import load

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

LLM_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_CACHE_DIR = Path.home() / ".cache" / "fastembed"

PASSAGE_WORDS = 80
PASSAGES_NEAR_EXTRACTION = 1
PASSAGES_NEAR_BLOCK = 1
EXAMPLES = 2
EXAMPLE_CHARS = 400
GRADES = np.array([1, 2, 3, 4, 5])

BLOCK_GUIDE = {
    "Требования истца": "what the plaintiff asks the court to award or order",
    "Аргументы истца": "the plaintiff's arguments and the grounds for the claims",
    "Аргументы ответчика": "the defendant's position and objections, or that the defendant gave none",
    "Оценка судом представленных сторонами доказательств": "how the court assessed the evidence the parties submitted",
    "Логические шаги в рассуждениях судьи и промежуточные выводы": "the chain of the judge's reasoning and intermediate conclusions",
    "Применимые в судебном деле нормы права": "the laws and articles the court applied",
    "Решение суда": "the court's final ruling (the operative part)",
}

SYSTEM_PROMPT = """You are an experienced Russian lawyer reviewing automatic extractions from court decisions.
You get the name of one block, passages of the decision that a search engine found relevant to it,
extractions of the same block from other decisions with the scores human experts gave them, and the
extracted text to grade. The passages are only part of the decision. The examples come from different
cases: use them to see how strict the experts are, not as content for this decision.

Grade how well the extracted text captures the block in THIS decision:
5 - complete and accurate: all essential content of the block, nothing invented, nothing from other blocks
4 - accurate with minor omissions or minor imprecision
3 - partly correct: important omissions, or noticeable content that belongs to another block
2 - mostly wrong, very incomplete, or largely about something else
1 - fails the task: wrong content, invented facts, or nothing usable

Answer with a single digit from 1 to 5 and nothing else."""


# ---------- embeddings ----------

def embed(embedder, texts, prefix):
    vectors = np.array(list(embedder.embed([prefix + text for text in texts], batch_size=32)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_extractions(embedder, df, name):
    path = OUTPUT_DIR / f"{name}_extraction_vectors.npy"
    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(df):
            return vectors
    print(f"embedding {len(df)} {name} extractions...")
    vectors = embed(embedder, list(df["extracted_text"]), "query: ")
    np.save(path, vectors)
    return vectors


def split_into_passages(text):
    words = text.split()
    return [" ".join(words[start : start + PASSAGE_WORDS]) for start in range(0, len(words), PASSAGE_WORDS)]


def embed_decisions(embedder, df):
    folder = OUTPUT_DIR / f"passage_vectors_{PASSAGE_WORDS}_words"
    folder.mkdir(exist_ok=True)
    cases = df.drop_duplicates("id_sud_resh")
    passages = {}
    started = time.time()
    for count, (case_id, source_text) in enumerate(zip(cases["id_sud_resh"], cases["source_text"]), 1):
        texts = split_into_passages(source_text)
        path = folder / f"{case_id}.npy"
        if path.exists():
            vectors = np.load(path)
        else:
            vectors = embed(embedder, texts, "passage: ")
            np.save(path, vectors)
        passages[case_id] = (texts, vectors)
        if count % 50 == 0 or count == len(cases):
            print(f"  {count}/{len(cases)} decisions, {(time.time() - started) / 60:.1f} min")
    return passages


# ---------- prompt ----------

def pick_passages(passage_texts, passage_vectors, extraction_vector, block_vector):
    near_extraction = np.argsort(-(passage_vectors @ extraction_vector))[:PASSAGES_NEAR_EXTRACTION]
    near_block = np.argsort(-(passage_vectors @ block_vector))[:PASSAGES_NEAR_BLOCK]
    chosen = sorted(set(near_extraction) | set(near_block))
    return [(round(100 * i / len(passage_texts)), passage_texts[i]) for i in chosen]


def pick_examples(train, train_vectors, block_name, case_id, extraction_vector):
    same_block_other_case = (train["block_name"] == block_name) & (train["id_sud_resh"] != case_id)
    candidates = np.flatnonzero(same_block_other_case.to_numpy())
    most_similar = candidates[np.argsort(-(train_vectors[candidates] @ extraction_vector))[:EXAMPLES]]
    return train.iloc[most_similar]


def build_prompt(block_name, passages, examples, extracted_text):
    lines = [f"BLOCK: {block_name} ({BLOCK_GUIDE[block_name]})", ""]
    lines.append("PASSAGES FROM THE DECISION (found by similarity search, in document order):")
    for position, text in passages:
        lines.append(f"[at {position}% of the decision] {text}")
    lines += ["", "EXPERT-SCORED EXTRACTIONS OF THE SAME BLOCK FROM OTHER DECISIONS:"]
    for score, text in zip(examples["expert_score_mean"], examples["extracted_text"]):
        shown = text if len(text) <= EXAMPLE_CHARS else text[:EXAMPLE_CHARS] + "..."
        lines.append(f"- expert score {score:.1f}, {len(text)} characters: {shown}")
    lines += ["", f"EXTRACTED TEXT TO GRADE ({len(extracted_text)} characters):", extracted_text, ""]
    lines.append("Grade (1-5):")
    return "\n".join(lines)


def make_prompts(df, vectors, train, train_vectors, passages, block_vectors):
    prompts = []
    for i in range(len(df)):
        row = df.iloc[i]
        passage_texts, passage_vectors = passages[row["id_sud_resh"]]
        block_vector = block_vectors[row["block_name"]]
        chosen = pick_passages(passage_texts, passage_vectors, vectors[i], block_vector)
        examples = pick_examples(train, train_vectors, row["block_name"], row["id_sud_resh"], vectors[i])
        prompts.append(build_prompt(row["block_name"], chosen, examples, row["extracted_text"]))
    return prompts


# ---------- LLM judge ----------

def load_judge():
    model, tokenizer = load(LLM_MODEL)
    digit_tokens = [tokenizer.encode(str(grade), add_special_tokens=False) for grade in GRADES]
    assert all(len(tokens) == 1 for tokens in digit_tokens), "each grade must be a single token"
    return model, tokenizer, [tokens[0] for tokens in digit_tokens]


def expected_grade(model, tokenizer, digit_tokens, prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    next_token_logits = model(mx.array(tokens)[None])[0, -1]
    probabilities = np.array(mx.softmax(next_token_logits.astype(mx.float32))[mx.array(digit_tokens)])
    digit_mass = probabilities.sum()
    return float((probabilities / digit_mass) @ GRADES), float(digit_mass)


def grade_all(judge, row_ids, prompts, path):
    model, tokenizer, digit_tokens = judge
    if not path.exists():
        path.write_text("row_id,grade,digit_mass\n")
    finished = set(pd.read_csv(path)["row_id"])
    todo = [(row_id, prompt) for row_id, prompt in zip(row_ids, prompts) if row_id not in finished]
    print(f"grading {len(todo)} rows ({len(row_ids) - len(todo)} already saved in {path.name})...")
    started = time.time()
    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        for count, (row_id, prompt) in enumerate(todo, 1):
            grade, digit_mass = expected_grade(model, tokenizer, digit_tokens, prompt)
            writer.writerow([row_id, grade, digit_mass])
            file.flush()
            if count % 50 == 0 or count == len(todo):
                print(f"  {count}/{len(todo)} rows, {(time.time() - started) / 60:.1f} min")
    grades = pd.read_csv(path).drop_duplicates("row_id").set_index("row_id").loc[list(row_ids)]
    print(f"  probability on the digits 1-5: mean {grades['digit_mass'].mean():.3f}, min {grades['digit_mass'].min():.3f}")
    return grades["grade"].to_numpy()


def rescale(predictions, target_mean, target_std):
    stretched = (predictions - predictions.mean()) / predictions.std() * target_std + target_mean
    return np.clip(stretched, 1, 5)


# ---------- metric ----------

def competition_score(y_true, y_pred):
    mae_part = 1 - np.mean(np.abs(y_true - y_pred)) / 4
    pearson = np.corrcoef(y_true, y_pred)[0, 1]
    covariance = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    ccc = 2 * covariance / (y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2)
    total = 0.15 * mae_part + 0.55 * ccc + 0.30 * pearson
    return {"total": total, "ccc": ccc, "pearson": pearson, "mae_part": mae_part}


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="LLM-only judge for LegalEval Extract")
    parser.add_argument("--cases", type=int, help="grade only the first N training cases and skip the test set")
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    y = train["expert_score_mean"].to_numpy()
    OUTPUT_DIR.mkdir(exist_ok=True)

    to_grade = train
    if args.cases:
        first_cases = train["id_sud_resh"].unique()[: args.cases]
        to_grade = train[train["id_sud_resh"].isin(first_cases)]

    print("loading the embedding model...")
    embedder = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(EMBEDDING_CACHE_DIR))
    train_vectors = embed_extractions(embedder, train, "train")
    block_vectors = dict(zip(BLOCK_GUIDE, embed(embedder, list(BLOCK_GUIDE), "query: ")))
    print("embedding decision passages (slow the first time, then read from outputs/)...")
    passages = embed_decisions(embedder, to_grade if args.cases else pd.concat([train, test]))

    print("loading the LLM...")
    judge = load_judge()

    print("training rows: examples always come from other cases, so this is a fair estimate")
    graded_rows = train["row_id"].isin(to_grade["row_id"]).to_numpy()
    prompts = make_prompts(to_grade, train_vectors[graded_rows], train, train_vectors, passages, block_vectors)
    (OUTPUT_DIR / "example_prompt.txt").write_text(SYSTEM_PROMPT + "\n\n" + prompts[0])
    grades = grade_all(judge, list(to_grade["row_id"]), prompts, OUTPUT_DIR / "grades_train.csv")
    y_graded = to_grade["expert_score_mean"].to_numpy()
    print(f"score on {len(to_grade)} training rows (grades stretched to the experts' mean and spread):")
    for name, value in competition_score(y_graded, rescale(grades, y.mean(), y.std())).items():
        print(f"  {name}: {value:.4f}")

    if args.cases:
        print("--cases given: skipping the test set")
        return

    test_vectors = embed_extractions(embedder, test, "test")
    prompts = make_prompts(test, test_vectors, train, train_vectors, passages, block_vectors)
    grades = grade_all(judge, list(test["row_id"]), prompts, OUTPUT_DIR / "grades_test.csv")
    submission = pd.DataFrame({"row_id": test["row_id"], "predicted_score": rescale(grades, y.mean(), y.std())})

    assert list(submission["row_id"]) == list(sample["row_id"]), "row_ids must match sample_submission.csv"
    assert submission["predicted_score"].between(1, 5).all(), "scores must be between 1 and 5"

    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"saved {OUTPUT_DIR / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
