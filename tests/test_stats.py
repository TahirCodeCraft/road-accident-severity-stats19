"""Tests for src.evaluation.stats."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import expected_calibration_error
from src.evaluation.stats import (
    ThresholdMetric,
    bootstrap_p_value,
    bootstrap_weights,
    delong_test,
    holm_correction,
    mcnemar_from_counts,
    mcnemar_test,
    paired_bootstrap,
    paired_bootstrap_many,
)


def _two_models(n: int, seed: int, gap: float = 0.5) -> tuple[np.ndarray, ...]:
    """Labels plus a stronger model A and weaker model B sharing noise (paired design)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    shared = rng.normal(size=n)
    p_a = 1 / (1 + np.exp(-(shared + (1.0 + gap) * y - 1.5)))
    p_b = 1 / (1 + np.exp(-(shared + 0.7 * rng.normal(size=n) + 1.0 * y - 1.5)))
    return y, p_a, p_b


# ----------------------------------------------------------------------------- bootstrap
def test_bootstrap_weights_stratified_preserve_class_counts() -> None:
    y = np.array([1] * 30 + [0] * 70)
    for w in bootstrap_weights(y, n_boot=5, seed=0, stratified=True):
        assert w.sum() == 100
        assert w[y == 1].sum() == 30
        assert np.all(w == np.round(w))


def test_identical_predictions_give_zero_delta_and_p_one() -> None:
    y, p_a, _ = _two_models(3000, seed=0)
    res = paired_bootstrap(y, p_a, p_a, "pr_auc", n_boot=200, seed=0)
    assert res.delta == 0 and res.ci_low == 0 and res.ci_high == 0
    assert res.p_value == 1.0


def test_bootstrap_point_estimates_match_sklearn() -> None:
    y, p_a, p_b = _two_models(4000, seed=1)
    res = paired_bootstrap_many(y, p_a, p_b, {"ap": "pr_auc", "auc": "roc_auc"}, n_boot=50)
    assert res["ap"].metric_a == pytest.approx(average_precision_score(y, p_a), abs=1e-12)
    assert res["auc"].metric_b == pytest.approx(roc_auc_score(y, p_b), abs=1e-12)


def test_bootstrap_ci_contains_true_delta() -> None:
    """Population delta estimated on a huge sample; CI from a modest sample should cover it."""
    y_big, a_big, b_big = _two_models(400_000, seed=123)
    true_delta = average_precision_score(y_big, a_big) - average_precision_score(y_big, b_big)
    y, p_a, p_b = _two_models(8000, seed=7)
    res = paired_bootstrap(y, p_a, p_b, "pr_auc", n_boot=1000, seed=0)
    assert res.ci_low <= true_delta <= res.ci_high
    assert res.ci_low > 0 and res.p_value < 0.01
    assert res.n_valid == 1000


def test_fast_path_matches_callable_path() -> None:
    y, p_a, p_b = _two_models(2000, seed=2)
    fast = paired_bootstrap(y, p_a, p_b, "pr_auc", n_boot=100, seed=3)
    slow = paired_bootstrap(y, p_a, p_b, average_precision_score, n_boot=100, seed=3)
    np.testing.assert_allclose(fast.deltas, slow.deltas, atol=1e-12)
    for name in ("roc_auc", "brier"):
        f = paired_bootstrap(y, p_a, p_b, name, n_boot=30, seed=4)
        fn = roc_auc_score if name == "roc_auc" else (lambda yy, pp: np.mean((pp - yy) ** 2))
        s = paired_bootstrap(y, p_a, p_b, fn, n_boot=30, seed=4)
        np.testing.assert_allclose(f.deltas, s.deltas, atol=1e-12)


def test_threshold_metric_matches_callable_and_supports_per_model_thresholds() -> None:
    y, p_a, p_b = _two_models(2000, seed=5)

    def f2_at(t: float) -> Callable[[np.ndarray, np.ndarray], float]:
        def fn(yy: np.ndarray, pp: np.ndarray) -> float:
            pred = pp >= t
            tp = np.sum(pred & (yy == 1))
            return 5 * tp / (5 * tp + 4 * np.sum(~pred & (yy == 1)) + np.sum(pred & (yy == 0)))
        return fn

    fast = paired_bootstrap(y, p_a, p_b, ThresholdMetric("f2", 0.2), n_boot=50, seed=1,
                            metric_fn_b=ThresholdMetric("f2", 0.3))
    slow = paired_bootstrap(y, p_a, p_b, f2_at(0.2), n_boot=50, seed=1, metric_fn_b=f2_at(0.3))
    np.testing.assert_allclose(fast.deltas, slow.deltas, atol=1e-12)
    assert fast.metric_a == pytest.approx(f2_at(0.2)(y, p_a))


def test_callable_ece_metric_runs() -> None:
    y, p_a, p_b = _two_models(2000, seed=6)
    res = paired_bootstrap(y, p_a, p_b, expected_calibration_error, n_boot=30, seed=0)
    assert np.isfinite(res.delta) and res.ci_low <= res.ci_high


def test_bootstrap_p_value_formula() -> None:
    assert bootstrap_p_value(np.array([1.0, 2.0, 3.0, -1.0])) == pytest.approx(2 * 2 / 5)
    assert bootstrap_p_value(np.zeros(10)) == 1.0


# ----------------------------------------------------------------------------- DeLong
def test_delong_identical_predictions() -> None:
    y, p_a, _ = _two_models(3000, seed=8)
    res = delong_test(y, p_a, p_a)
    assert res.p_value == pytest.approx(1.0)
    assert res.delta == 0.0


def test_delong_aucs_match_sklearn_and_detect_difference() -> None:
    y, p_a, p_b = _two_models(5000, seed=9)
    res = delong_test(y, p_a, np.round(p_b, 2))
    assert res.auc_a == pytest.approx(roc_auc_score(y, p_a), abs=1e-12)
    assert res.auc_b == pytest.approx(roc_auc_score(y, np.round(p_b, 2)), abs=1e-12)
    assert res.p_value < 1e-3 and res.ci_low > 0


def test_delong_variance_matches_reference_example() -> None:
    """Small hand-checkable case: compare with an O(m*n) direct DeLong implementation."""
    rng = np.random.default_rng(10)
    y = np.array([1] * 12 + [0] * 18)
    a = rng.normal(size=30) + y
    b = np.round(a + rng.normal(scale=0.8, size=30), 1)
    pos, neg = y == 1, y == 0

    def psi(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])

    comps = []
    for s in (a, b):
        k = psi(s[pos], s[neg])
        comps.append((k.mean(), k.mean(axis=1), k.mean(axis=0)))
    s10 = np.cov(np.vstack([comps[0][1], comps[1][1]]))
    s01 = np.cov(np.vstack([comps[0][2], comps[1][2]]))
    cov = s10 / pos.sum() + s01 / neg.sum()
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    res = delong_test(y, 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b)))
    assert res.auc_a == pytest.approx(comps[0][0])
    assert res.se == pytest.approx(se)
    assert res.p_value == pytest.approx(2 * sps.norm.sf(abs((comps[0][0] - comps[1][0]) / se)))


# ----------------------------------------------------------------------------- McNemar
def test_mcnemar_exact_known_counts() -> None:
    res = mcnemar_from_counts(10, 2)
    assert res.method == "exact"
    assert res.p_value == pytest.approx(158 / 4096)


def test_mcnemar_chi2_known_counts() -> None:
    res = mcnemar_from_counts(59, 33)
    assert res.method == "chi2"
    assert res.statistic == pytest.approx(625 / 92)
    assert res.p_value == pytest.approx(sps.chi2.sf(625 / 92, 1))
    assert res.p_value == pytest.approx(0.00915, abs=1e-4)


def test_mcnemar_from_predictions() -> None:
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    pred_a = np.array([1, 1, 1, 0, 0, 0, 0, 1])  # 6 correct
    pred_b = np.array([0, 0, 1, 0, 1, 0, 0, 1])  # A right / B wrong at 0,1,4
    res = mcnemar_test(y, pred_a, pred_b)
    assert (res.n01, res.n10) == (3, 0)
    assert res.p_value == pytest.approx(0.25)
    assert mcnemar_test(y, pred_a, pred_a).p_value == 1.0


# ----------------------------------------------------------------------------- Holm
def test_holm_hand_computed() -> None:
    adjusted = holm_correction([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm_correction([0.5, 0.9]), [1.0, 1.0])


def test_holm_matches_statsmodels_if_available() -> None:
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    p = np.random.default_rng(0).uniform(0, 0.2, size=12)
    expected = multitest.multipletests(p, method="holm")[1]
    np.testing.assert_allclose(holm_correction(p), expected)
