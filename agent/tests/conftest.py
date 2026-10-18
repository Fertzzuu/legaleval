import numpy as np
import pandas as pd
import pytest
from legaleval_gbm.data import BLOCKS
from legaleval_gbm.text import Lemmatizer

from legaleval_agent.judge_model import JudgeModel, train_judge

WORDS = "истец ответчик суд договор долг взыскать иск решение доказательство статья закон 100 2024 15".split()
INVENTED = "выдумка фантазия ошибка неверно".split()


def _synthetic_train(n_cases: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_cases):
        source = " ".join(rng.choice(WORDS, 400))
        source_words = source.split()
        for block in BLOCKS:
            length = int(rng.integers(3, 60))
            start = int(rng.integers(0, len(source_words) - length))
            extracted = " ".join(source_words[start : start + length])
            if rng.random() < 0.3:
                extracted += " " + " ".join(rng.choice(INVENTED, int(rng.integers(1, 4))))
            rows.append(
                {
                    "row_id": f"C{c}|{block}",
                    "id_sud_resh": f"C{c}",
                    "block_name": block,
                    "extracted_text": extracted,
                    "source_text": source,
                    "expert_score_mean": float(np.clip(1 + length / 15 + rng.normal(0, 0.3), 1, 5)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def lemmatizer() -> Lemmatizer:
    return Lemmatizer()


@pytest.fixture(scope="session")
def synthetic_train() -> pd.DataFrame:
    return _synthetic_train()


@pytest.fixture(scope="session")
def judge(synthetic_train: pd.DataFrame, lemmatizer: Lemmatizer) -> JudgeModel:
    return train_judge(synthetic_train, lemmatizer, seeds=[0], folds=3)
