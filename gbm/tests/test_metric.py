import numpy as np
import pytest

from legaleval_gbm.metric import competition_score, lins_ccc, mae_norm, pearson, rescale


def test_perfect_prediction_scores_one():
    y = np.array([1.0, 3.0, 4.0, 5.0])
    score = competition_score(y, y)
    assert score.total == pytest.approx(1.0)
    assert score.mae_norm == pytest.approx(1.0)
    assert score.ccc == pytest.approx(1.0)
    assert score.pearson == pytest.approx(1.0)


def test_hand_computed_example():
    y = np.array([2.0, 4.0, 4.0, 5.0])
    p = np.array([3.0, 3.0, 4.0, 4.0])
    assert mae_norm(y, p) == pytest.approx(1 - 0.75 / 4)
    assert lins_ccc(y, p) == pytest.approx(2 * 0.375 / (1.1875 + 0.25 + 0.0625))
    assert pearson(y, p) == pytest.approx(0.375 / np.sqrt(1.1875 * 0.25))


def test_shift_keeps_pearson_but_lowers_ccc():
    y = np.array([1.0, 2.0, 4.0, 5.0])
    shifted = y + 1.0
    assert pearson(y, shifted) == pytest.approx(1.0)
    assert lins_ccc(y, shifted) < 1.0


def test_shrunk_predictions_lose_ccc_until_rescaled():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    shrunk = 3.0 + (y - 3.0) * 0.2
    assert lins_ccc(y, shrunk) < 0.5
    assert lins_ccc(y, rescale(shrunk, y.mean(), y.std())) == pytest.approx(1.0)


def test_rescale_clips_to_scale():
    stretched = rescale(np.array([0.0, 1.0, 2.0]), target_mean=4.5, target_std=2.0)
    assert stretched.min() >= 1.0
    assert stretched.max() == pytest.approx(5.0)


def test_rescale_rejects_constant_predictions():
    with pytest.raises(ValueError):
        rescale(np.array([3.0, 3.0]), target_mean=4.0, target_std=1.0)
