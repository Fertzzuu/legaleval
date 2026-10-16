from pathlib import Path
from typing import Literal

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

TARGET = "expert_score_mean"
ROW_ID = "row_id"
CASE_ID = "id_sud_resh"
BLOCK = "block_name"
EXTRACTED = "extracted_text"
SOURCE = "source_text"

BLOCKS = (
    "Требования истца",
    "Аргументы истца",
    "Аргументы ответчика",
    "Оценка судом представленных сторонами доказательств",
    "Логические шаги в рассуждениях судьи и промежуточные выводы",
    "Применимые в судебном деле нормы права",
    "Решение суда",
)

BLOCK_LABELS = dict(
    zip(
        BLOCKS,
        (
            "plaintiff claims",
            "plaintiff arguments",
            "defendant arguments",
            "evidence assessment",
            "reasoning steps",
            "applicable law",
            "decision",
        ),
        strict=True,
    )
)

_INPUT_COLUMNS = (ROW_ID, CASE_ID, BLOCK, EXTRACTED, SOURCE)


def load_split(data_dir: Path, split: Literal["train", "test"]) -> pd.DataFrame:
    df = pd.read_csv(data_dir / f"{split}.csv")
    required = _INPUT_COLUMNS + ((TARGET,) if split == "train" else ())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split}.csv is missing columns {missing}")
    unknown = set(df[BLOCK]) - set(BLOCKS)
    if unknown:
        raise ValueError(f"{split}.csv has unknown block names {sorted(unknown)}")
    if df[ROW_ID].duplicated().any():
        raise ValueError(f"{split}.csv has duplicate {ROW_ID} values")
    return df


def load_sample_submission(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "sample_submission.csv")
