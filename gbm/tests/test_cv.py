import numpy as np
import pandas as pd
import pytest
from sklearn.inspection import permutation_importance

from legaleval_gbm.cv import GbmRidgeBlend, gbm, ridge
from legaleval_gbm.data import BLOCKS


@pytest.fixture
def data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 140
    X = pd.DataFrame(
        {
            "block_name": pd.Categorical(rng.choice(BLOCKS, n), categories=BLOCKS),
            "a": rng.random(n),
            "b": rng.random(n),
        }
    )
    X.loc[::9, "b"] = np.nan
    y = 1.0 + 3.0 * X["a"].to_numpy() + (X["block_name"] == BLOCKS[6]).to_numpy() + rng.normal(0, 0.1, n)
    return X, y


def test_ridge_handles_missing_values_and_blocks(data):
    X, y = data
    predictions = ridge(0).fit(X, y).predict(X)
    assert np.isfinite(predictions).all()
    assert np.corrcoef(predictions, y)[0, 1] > 0.9


def test_blend_is_the_weighted_average_of_its_parts(data):
    X, y = data
    blend = GbmRidgeBlend(seed=0, gbm_weight=0.6, gbm_trees=50).fit(X, y)
    expected = 0.6 * gbm(0).set_params(max_iter=50).fit(X, y).predict(X) + 0.4 * ridge(0).fit(X, y).predict(X)
    assert blend.predict(X) == pytest.approx(expected)


def test_blend_works_with_permutation_importance(data):
    X, y = data
    blend = GbmRidgeBlend(gbm_trees=50).fit(X, y)
    result = permutation_importance(
        blend, X, y, scoring="neg_mean_squared_error", n_repeats=2, random_state=0
    )
    assert result.importances_mean[1] > result.importances_mean[2]
