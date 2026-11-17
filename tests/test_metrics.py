"""Tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.metrics import (
    average_precision_fast,
    binary_metrics,
    calibration_bin_ids,
    confusion_counts,
    expected_calibration_error,
    multiclass_metrics,
    reliability_curve,
    roc_auc_fast,
    sort_scores,
    weighted_average_precision,
    weighted_roc_auc,
)


@pytest.fixture()
def toy() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic data with ~25% prevalence and rounded (tied) probabilities."""
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.25).astype(int)
    p = np.round(1.0 / (1.0 + np.exp(-(rng.normal(size=n) + 1.2 * y - 1.0))), 2)
    return y, p


def test_binary_metrics_match_sklearn(toy: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = toy
    m = binary_metrics(y, p, threshold=0.3)
    pred = (p >= 0.3).astype(int)
    assert m.precision == pytest.approx(precision_score(y, pred))
    assert m.recall == pytest.approx(recall_score(y, pred))
    assert m.f1 == pytest.approx(f1_score(y, pred))
    assert m.f2 == pytest.approx(fbeta_score(y, pred, beta=2))
    assert m.pr_auc == pytest.approx(average_precision_score(y, p))
    assert m.roc_auc == pytest.approx(roc_auc_score(y, p))
    assert m.brier == pytest.approx(brier_score_loss(y, p))
    assert m.log_loss == pytest.approx(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)))
    assert m.tp + m.fp + m.tn + m.fn == y.size
    assert m.pred_pos_rate == pytest.approx(pred.mean())
    assert m.balanced_accuracy == pytest.approx(0.5 * (m.recall + m.specificity))


def test_expected_cost_default_ratio() -> None:
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.1])  # at t=0.5: 1 FN, 1 FP
    m = binary_metrics(y, p, threshold=0.5)
    assert (m.fn, m.fp) == (1, 1)
    assert m.expected_cost == pytest.approx((5 * 1 + 1 * 1) / 5)
    assert binary_metrics(y, p, 0.5, fn_cost=2, fp_cost=3).expected_cost == pytest.approx(1.0)


def test_confusion_counts() -> None:
    c = confusion_counts([1, 0, 1, 0, 1], [1, 1, 0, 0, 1])
    assert (c.tp, c.fp, c.tn, c.fn) == (2, 1, 1, 1)


def test_fast_ranking_metrics_match_sklearn(toy: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = toy
    assert average_precision_fast(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-12)
    assert roc_auc_fast(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_weighted_ranking_metrics_match_sklearn_sample_weight(
    toy: tuple[np.ndarray, np.ndarray],
) -> None:
    y, p = toy
    w = np.random.default_rng(1).integers(0, 4, size=y.size).astype(float)
    scores = sort_scores(y, p)
    w_sorted = w[scores.order]
    expected_ap = average_precision_score(y, p, sample_weight=w)
    assert weighted_average_precision(scores, w_sorted) == pytest.approx(expected_ap, abs=1e-12)
    expected_auc = roc_auc_score(y, p, sample_weight=w)
    assert weighted_roc_auc(scores, w_sorted) == pytest.approx(expected_auc, abs=1e-12)


@pytest.mark.parametrize("strategy", ["quantile", "uniform"])
def test_ece_zero_for_perfect_calibration(strategy: str) -> None:
    levels = np.array([0.1, 0.25, 0.5, 0.8])
    p = np.repeat(levels, 100)
    y = np.concatenate([np.r_[np.ones(int(100 * v)), np.zeros(100 - int(100 * v))] for v in levels])
    assert expected_calibration_error(y, p, n_bins=15, strategy=strategy) == pytest.approx(0.0)


def test_ece_detects_miscalibration() -> None:
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.05, 0.5, size=20000)
    y = (rng.random(true_p.size) < true_p).astype(int)
    assert expected_calibration_error(y, true_p) < 0.02
    assert expected_calibration_error(y, np.clip(true_p + 0.2, 0, 1)) > 0.15


def test_equal_mass_bins_are_balanced() -> None:
    p = np.random.default_rng(4).random(1500)
    ids, edges = calibration_bin_ids(p, n_bins=15, strategy="quantile")
    counts = np.bincount(ids)
    assert counts.size == 15 and len(edges) == 16
    assert counts.min() >= 99 and counts.max() <= 101


def test_reliability_curve_shapes(toy: tuple[np.ndarray, np.ndarray]) -> None:
    y, p = toy
    curve = reliability_curve(y, p, n_bins=10, strategy="uniform")
    assert curve.count.sum() == y.size
    assert curve.mean_pred.shape == curve.frac_pos.shape == curve.count.shape
    assert np.all(np.diff(curve.mean_pred) > 0)


def test_single_class_gives_nan_ranking_metrics() -> None:
    m = binary_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10))
    assert np.isnan(m.pr_auc) and np.isnan(m.roc_auc)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        binary_metrics([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.1, 1.2])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.1])


def test_multiclass_metrics() -> None:
    y = np.array([0, 0, 1, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0, 2])
    m = multiclass_metrics(y, pred, labels=[0, 1, 2])
    assert m.macro_f1 == pytest.approx(f1_score(y, pred, average="macro"))
    assert m.per_class_recall == pytest.approx({0: 0.5, 1: 1.0, 2: 2 / 3})
    assert m.balanced_accuracy == pytest.approx(np.mean([0.5, 1.0, 2 / 3]))
    assert m.confusion.sum() == y.size
    assert "recall_2" in m.to_dict()
