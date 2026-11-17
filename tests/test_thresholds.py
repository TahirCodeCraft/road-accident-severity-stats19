"""Tests for src.evaluation.thresholds."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.metrics import fbeta_score, precision_score, recall_score

from src.evaluation.thresholds import (
    ThresholdConfig,
    ThresholdPolicy,
    choose_threshold,
    cost_threshold,
    max_fbeta_threshold,
    min_cost_threshold,
    precision_at_recall_threshold,
    threshold_curve,
)


@pytest.fixture()
def data() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic set with ties so brute force stays cheap."""
    rng = np.random.default_rng(7)
    n = 1500
    y = (rng.random(n) < 0.25).astype(int)
    p = np.round(1.0 / (1.0 + np.exp(-(rng.normal(size=n) + 1.0 * y - 1.0))), 3)
    return y, p


def _brute_force(
    y: np.ndarray, p: np.ndarray, score_fn: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    """Return the best threshold by brute force (ties -> highest threshold)."""
    best_t, best_s = None, -np.inf
    for t in np.unique(p):
        s = score_fn(y, (p >= t).astype(int))
        if s >= best_s:
            best_t, best_s = t, s
    return float(best_t)


def test_threshold_curve_matches_direct_counts(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = data
    ts = np.array([0.0, 0.2, 0.31, 0.5, 1.01])
    curve = threshold_curve(y, p, ts)
    for i, t in enumerate(ts):
        pred = p >= t
        assert curve.tp[i] == np.sum(pred & (y == 1))
        assert curve.fp[i] == np.sum(pred & (y == 0))
        assert curve.fn[i] == np.sum(~pred & (y == 1))
        assert curve.tn[i] == np.sum(~pred & (y == 0))


def test_max_fbeta_matches_brute_force(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = data
    expected = _brute_force(y, p, lambda yt, yp: fbeta_score(yt, yp, beta=2, zero_division=0))
    assert max_fbeta_threshold(y, p, beta=2.0) == pytest.approx(expected)


def test_max_fbeta_grid_is_close(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = data
    t_exact = max_fbeta_threshold(y, p)
    t_grid = max_fbeta_threshold(y, p, n_grid=200)
    f_exact = fbeta_score(y, p >= t_exact, beta=2)
    f_grid = fbeta_score(y, p >= t_grid, beta=2)
    assert f_grid == pytest.approx(f_exact, abs=0.01)


def test_precision_at_recall(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = data

    def score(yt: np.ndarray, yp: np.ndarray) -> float:
        if recall_score(yt, yp) < 0.8:
            return -1.0
        return precision_score(yt, yp, zero_division=0)

    t = precision_at_recall_threshold(y, p, min_recall=0.8)
    assert recall_score(y, p >= t) >= 0.8
    assert t == pytest.approx(_brute_force(y, p, score))


def test_precision_at_recall_infeasible_raises() -> None:
    with pytest.raises(ValueError):
        precision_at_recall_threshold([0, 0, 0], [0.1, 0.2, 0.3], min_recall=0.8)


def test_cost_threshold() -> None:
    assert cost_threshold() == pytest.approx(1 / 6)
    assert cost_threshold(fn_cost=1, fp_cost=1) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cost_threshold(fn_cost=0)


def test_min_cost_threshold_near_bayes_for_calibrated_probs() -> None:
    rng = np.random.default_rng(11)
    p = rng.beta(1.2, 3.0, size=200_000)
    y = (rng.random(p.size) < p).astype(int)
    assert min_cost_threshold(y, p, n_grid=400) == pytest.approx(1 / 6, abs=0.03)


@pytest.mark.parametrize("policy", list(ThresholdPolicy))
def test_dispatcher_all_policies(
    data: tuple[np.ndarray, np.ndarray], policy: ThresholdPolicy
) -> None:
    y, p = data
    res = choose_threshold(y, p, ThresholdConfig(policy=policy))
    assert res.policy == policy.value
    assert 0.0 <= res.threshold <= 1.0
    assert 0.0 <= res.recall <= 1.0 and 0.0 <= res.precision <= 1.0
    if policy is ThresholdPolicy.COST_BAYES:
        assert res.threshold == pytest.approx(1 / 6)
    if policy is ThresholdPolicy.MAX_FBETA:
        assert res.fbeta == pytest.approx(fbeta_score(y, p >= res.threshold, beta=2))


def test_dispatcher_accepts_string_policy(data: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = data
    config = ThresholdConfig(policy="fixed", fixed=0.4)  # type: ignore[arg-type]
    res = choose_threshold(y, p, config)
    assert res.threshold == 0.4
