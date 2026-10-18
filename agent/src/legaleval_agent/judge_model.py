import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from legaleval_gbm.cv import GbmRidgeBlend, gbm_ridge_blend, run_cv
from legaleval_gbm.data import BLOCK, BLOCKS, CASE_ID, EXTRACTED, ROW_ID, SOURCE, TARGET
from legaleval_gbm.features import FEATURE_SETS, build_features
from legaleval_gbm.metric import SCALE_MAX, SCALE_MIN, Score
from legaleval_gbm.text import Lemmatizer

COLUMNS = list(FEATURE_SETS["lexical"])
REASON_FEATURES = ("ext_chars", "copy_ratio", "trigram_copy_ratio", "novel_words")
LOW, MID, HIGH = 0.10, 0.50, 0.90


@dataclass(frozen=True)
class Range:
    low: float
    typical: float
    high: float


@dataclass(frozen=True)
class Verdict:
    score: float
    typical_score_for_block: float
    reasons: list[str]


@dataclass(frozen=True)
class OutOfRange:
    reason: str


@dataclass(frozen=True)
class JudgeModel:
    model: GbmRidgeBlend
    prediction_mean: float
    prediction_std: float
    target_mean: float
    target_std: float
    block_scores: dict[str, float]
    ranges: dict[str, dict[str, Range]]
    max_chars: dict[str, int]
    cv_scores: list[Score]

    def rescale(self, raw: np.ndarray) -> np.ndarray:
        stretched = (raw - self.prediction_mean) / self.prediction_std * self.target_std + self.target_mean
        return np.clip(stretched, SCALE_MIN, SCALE_MAX)

    def score_features(self, features: pd.DataFrame) -> np.ndarray:
        return self.rescale(self.model.predict(features[COLUMNS]))

    def judge(
        self, decision: str, block_name: str, extracted: str, lemmatizer: Lemmatizer
    ) -> Verdict | OutOfRange:
        longest = self.max_chars[block_name]
        if len(extracted) > longest:
            return OutOfRange(
                f"cannot score: {len(extracted)} chars is longer than any '{block_name}' extraction the "
                f"judge was trained on (longest {longest}); shorten it to the essential content"
            )
        features = build_features(single_row(decision, block_name, extracted), lemmatizer)
        values = {f: float(features[f].iloc[0]) for f in REASON_FEATURES}
        return Verdict(
            score=round(float(self.score_features(features)[0]), 2),
            typical_score_for_block=round(self.block_scores[block_name], 2),
            reasons=reasons(values, block_name, self.ranges[block_name]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "JudgeModel":
        loaded = joblib.load(path)
        if not isinstance(loaded, JudgeModel):
            raise TypeError(f"{path} does not contain a JudgeModel")
        return loaded


def single_row(decision: str, block_name: str, extracted: str) -> pd.DataFrame:
    if block_name not in BLOCKS:
        raise ValueError(f"unknown block {block_name!r}")
    return pd.DataFrame(
        {
            ROW_ID: [f"agent|{block_name}"],
            CASE_ID: ["agent"],
            BLOCK: [block_name],
            EXTRACTED: [extracted],
            SOURCE: [decision],
        }
    )


def reasons(values: dict[str, float], block_name: str, ranges: dict[str, Range]) -> list[str]:
    found = []
    chars, length = values["ext_chars"], ranges["ext_chars"]
    if chars < length.low:
        found.append(
            f"shorter than 90% of '{block_name}' extractions "
            f"({int(chars)} chars, typical {int(length.typical)})"
        )
    elif chars > length.high:
        found.append(
            f"longer than 90% of '{block_name}' extractions "
            f"({int(chars)} chars, typical {int(length.typical)})"
        )
    copy = values["copy_ratio"]
    if not math.isnan(copy) and copy < ranges["copy_ratio"].low:
        found.append(
            f"only {copy:.0%} of its words appear in the decision "
            f"(typical {ranges['copy_ratio'].typical:.0%})"
        )
    trigram = values["trigram_copy_ratio"]
    if not math.isnan(trigram) and trigram < ranges["trigram_copy_ratio"].low:
        found.append(
            f"only {trigram:.0%} of its three-word phrases appear verbatim in the decision "
            f"(typical {ranges['trigram_copy_ratio'].typical:.0%})"
        )
    novel = values["novel_words"]
    if novel > ranges["novel_words"].high:
        found.append(
            f"{int(novel)} of its words are not in the decision "
            f"(typical {int(ranges['novel_words'].typical)})"
        )
    return found


def feature_ranges(features: pd.DataFrame) -> dict[str, dict[str, Range]]:
    ranges: dict[str, dict[str, Range]] = {}
    for block_name, part in features.groupby(BLOCK, observed=True):
        quantiles = part[list(REASON_FEATURES)].quantile([LOW, MID, HIGH])
        ranges[str(block_name)] = {
            f: Range(*(float(quantiles.loc[q, f]) for q in (LOW, MID, HIGH))) for f in REASON_FEATURES
        }
    return ranges


def train_judge(train: pd.DataFrame, lemmatizer: Lemmatizer, seeds: list[int], folds: int) -> JudgeModel:
    features = build_features(train, lemmatizer)
    X, y = features[COLUMNS], train[TARGET].to_numpy()
    cv = run_cv("judge", gbm_ridge_blend, X, y, train[CASE_ID], seeds, folds)
    model = gbm_ridge_blend(seeds[0]).fit(X, y)
    return JudgeModel(
        model=model,
        prediction_mean=float(cv.oof_raw.mean(axis=1).mean()),
        prediction_std=float(cv.oof_raw.std(axis=1).mean()),
        target_mean=float(y.mean()),
        target_std=float(y.std()),
        block_scores={str(b): float(v) for b, v in train.groupby(BLOCK)[TARGET].mean().items()},
        ranges=feature_ranges(features),
        max_chars={
            str(b): int(v)
            for b, v in train.groupby(BLOCK)[EXTRACTED].apply(lambda s: s.str.len().max()).items()
        },
        cv_scores=cv.rescaled,
    )
