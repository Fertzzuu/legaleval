from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

SCALE_MIN = 1.0
SCALE_MAX = 5.0

WEIGHT_MAE = 0.15
WEIGHT_CCC = 0.55
WEIGHT_PEARSON = 0.30


@dataclass(frozen=True)
class Score:
    total: float
    mae_norm: float
    ccc: float
    pearson: float


def mae_norm(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(1.0 - np.mean(np.abs(y - p)) / (SCALE_MAX - SCALE_MIN))


def lins_ccc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    covariance = np.mean((y - y.mean()) * (p - p.mean()))
    return float(2.0 * covariance / (y.var() + p.var() + (y.mean() - p.mean()) ** 2))


def pearson(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.corrcoef(y, p)[0, 1])


def competition_score(y_true: ArrayLike, y_pred: ArrayLike) -> Score:
    m, c, r = mae_norm(y_true, y_pred), lins_ccc(y_true, y_pred), pearson(y_true, y_pred)
    return Score(
        total=WEIGHT_MAE * m + WEIGHT_CCC * c + WEIGHT_PEARSON * r,
        mae_norm=m,
        ccc=c,
        pearson=r,
    )


def rescale(y_pred: ArrayLike, target_mean: float, target_std: float) -> np.ndarray:
    p = np.asarray(y_pred, dtype=float)
    spread = p.std()
    if spread == 0.0:
        raise ValueError("cannot rescale constant predictions")
    stretched = (p - p.mean()) / spread * target_std + target_mean
    return np.clip(stretched, SCALE_MIN, SCALE_MAX)
