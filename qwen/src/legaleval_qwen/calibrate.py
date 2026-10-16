from typing import Self

import numpy as np
import pandas as pd
from legaleval_gbm.data import BLOCK, BLOCKS
from legaleval_gbm.metric import SCALE_MAX, SCALE_MIN
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

GRADE = "llm_grade"


class RawGrade:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[GRADE].to_numpy(dtype=float)


class IsotonicGrade:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        self.model_ = IsotonicRegression(y_min=SCALE_MIN, y_max=SCALE_MAX, out_of_bounds="clip")
        self.model_.fit(X[GRADE].to_numpy(dtype=float), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[GRADE].to_numpy(dtype=float))


class LinearGradeByBlock:
    def _design(self, X: pd.DataFrame) -> np.ndarray:
        blocks = pd.get_dummies(pd.Categorical(X[BLOCK], categories=BLOCKS), drop_first=True, dtype=float)
        return np.column_stack([X[GRADE].to_numpy(dtype=float), blocks.to_numpy()])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        self.model_ = LinearRegression().fit(self._design(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._design(X))


def raw_grade(seed: int) -> RawGrade:
    return RawGrade()


def isotonic_grade(seed: int) -> IsotonicGrade:
    return IsotonicGrade()


def linear_grade_by_block(seed: int) -> LinearGradeByBlock:
    return LinearGradeByBlock()
