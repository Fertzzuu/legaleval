from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from legaleval_gbm.data import BLOCK
from legaleval_gbm.metric import Score, competition_score, rescale


class Regressor(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


ModelFactory = Callable[[int], Regressor]


class BlockPriorRegressor:
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Self:
        self.means_ = pd.Series(y, index=X.index).groupby(X[BLOCK], observed=True).mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = X[BLOCK].map(self.means_).astype(float)
        if predictions.isna().any():
            raise ValueError("block absent from the training fold")
        return predictions.to_numpy()


def block_prior(seed: int) -> BlockPriorRegressor:
    return BlockPriorRegressor()


def gbm(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        categorical_features="from_dtype",
        random_state=seed,
    )


@dataclass(frozen=True)
class CvResult:
    name: str
    oof_raw: np.ndarray
    oof_rescaled: np.ndarray
    raw: list[Score]
    rescaled: list[Score]


def run_cv(
    name: str,
    make_model: ModelFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    seeds: Sequence[int],
    n_splits: int,
) -> CvResult:
    oof_raw = np.zeros((len(seeds), len(y)))
    for i, seed in enumerate(seeds):
        folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fit_idx, held_idx in folds.split(X, y, groups):
            model = make_model(seed).fit(X.iloc[fit_idx], y[fit_idx])
            oof_raw[i, held_idx] = model.predict(X.iloc[held_idx])
    oof_rescaled = np.vstack([rescale(p, y.mean(), y.std()) for p in oof_raw])
    return CvResult(
        name=name,
        oof_raw=oof_raw,
        oof_rescaled=oof_rescaled,
        raw=[competition_score(y, p) for p in oof_raw],
        rescaled=[competition_score(y, p) for p in oof_rescaled],
    )


def cv_permutation_importance(
    make_model: ModelFactory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    seed: int,
    n_splits: int,
    n_repeats: int,
) -> pd.DataFrame:
    folds = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_fold = []
    for fit_idx, held_idx in folds.split(X, y, groups):
        model = make_model(seed).fit(X.iloc[fit_idx], y[fit_idx])
        result = permutation_importance(
            model,
            X.iloc[held_idx],
            y[held_idx],
            scoring="neg_mean_squared_error",
            n_repeats=n_repeats,
            random_state=seed,
        )
        per_fold.append(result.importances_mean)
    importances = np.vstack(per_fold)
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "mse_increase": importances.mean(axis=0),
                "fold_std": importances.std(axis=0),
            }
        )
        .sort_values("mse_increase", ascending=False)
        .reset_index(drop=True)
    )


def fit_predict(
    make_model: ModelFactory,
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    seeds: Sequence[int],
) -> np.ndarray:
    predictions = np.mean([make_model(s).fit(X_train, y).predict(X_test) for s in seeds], axis=0)
    return rescale(predictions, y.mean(), y.std())
