"""Decision-threshold policies fitted on a calibration split (CAL) and frozen before TEST.

Convention: a sample is predicted positive when ``y_prob >= threshold``. Candidate thresholds are
the unique predicted probabilities (exact search, O(n log n) via one sort + cumulative counts) or,
optionally, a quantile grid of them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.evaluation._checks import check_binary_inputs
from src.evaluation.metrics import DEFAULT_FN_COST, DEFAULT_FP_COST, fbeta_from_counts, safe_divide

logger = logging.getLogger(__name__)


class ThresholdPolicy(str, Enum):
    """Supported threshold-selection policies."""

    FIXED = "fixed"
    MAX_FBETA = "max_fbeta"
    PRECISION_AT_RECALL = "precision_at_recall"
    COST_BAYES = "cost_bayes"
    COST_EMPIRICAL = "cost_empirical"


@dataclass(frozen=True)
class ThresholdConfig:
    """Parameters for :func:`choose_threshold`.

    Attributes:
        policy: Which policy to apply.
        fixed: Threshold used by ``FIXED``.
        beta: F-beta weight used by ``MAX_FBETA``.
        min_recall: Recall constraint used by ``PRECISION_AT_RECALL``.
        fn_cost: False-negative cost used by the cost policies.
        fp_cost: False-positive cost used by the cost policies.
        n_grid: If set, restrict candidates to this many probability quantiles.
    """

    policy: ThresholdPolicy = ThresholdPolicy.MAX_FBETA
    fixed: float = 0.5
    beta: float = 2.0
    min_recall: float = 0.80
    fn_cost: float = DEFAULT_FN_COST
    fp_cost: float = DEFAULT_FP_COST
    n_grid: int | None = None


@dataclass(frozen=True)
class ThresholdResult:
    """A chosen threshold and its operating point on the data it was chosen on."""

    policy: str
    threshold: float
    precision: float
    recall: float
    fbeta: float
    beta: float
    pred_pos_rate: float

    def to_dict(self) -> dict[str, float | str]:
        """Return the result as a plain dict (for manifests / JSON)."""
        return asdict(self)


@dataclass(frozen=True)
class ThresholdCurve:
    """Confusion counts at each candidate threshold (``predict y_prob >= threshold``)."""

    thresholds: NDArray[np.float64]
    tp: NDArray[np.float64]
    fp: NDArray[np.float64]
    fn: NDArray[np.float64]
    tn: NDArray[np.float64]


def candidate_thresholds(y_prob: ArrayLike, n_grid: int | None = None) -> NDArray[np.float64]:
    """Candidate thresholds: all unique probabilities, or a quantile grid of them.

    Args:
        y_prob: Probabilities.
        n_grid: Number of quantile grid points, or ``None`` for all unique values.

    Returns:
        Sorted unique candidate thresholds.
    """
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    cands = np.unique(p) if n_grid is None else np.unique(
        np.quantile(p, np.linspace(0.0, 1.0, n_grid)))
    # A cutoff strictly above every score ("predict nobody positive") is also a valid decision.
    return np.append(cands, np.nextafter(cands[-1], np.inf))


def threshold_curve(
    y_true: ArrayLike, y_prob: ArrayLike, thresholds: ArrayLike | None = None
) -> ThresholdCurve:
    """Confusion counts for many thresholds at once using one sort and a cumulative sum.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        thresholds: Thresholds to evaluate (default: all unique probabilities).

    Returns:
        Counts at each threshold.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    t = candidate_thresholds(p) if thresholds is None else np.asarray(thresholds, np.float64)
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    cum_pos = np.concatenate([[0.0], np.cumsum(y[order], dtype=np.float64)])
    n_below = np.searchsorted(p_sorted, t, side="left")
    total_pos = cum_pos[-1]
    tp = total_pos - cum_pos[n_below]
    pred_pos = (p.size - n_below).astype(np.float64)
    fp = pred_pos - tp
    fn = total_pos - tp
    tn = (p.size - total_pos) - fp
    return ThresholdCurve(thresholds=t, tp=tp, fp=fp, fn=fn, tn=tn)


def max_fbeta_threshold(
    y_true: ArrayLike, y_prob: ArrayLike, beta: float = 2.0, n_grid: int | None = None
) -> float:
    """Threshold maximising F-beta (ties broken towards the higher threshold).

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        beta: F-beta weight (2 = recall-oriented).
        n_grid: Optional quantile-grid size instead of exhaustive search.

    Returns:
        The F-beta-optimal threshold.
    """
    curve = threshold_curve(y_true, y_prob, candidate_thresholds(y_prob, n_grid))
    scores = fbeta_from_counts(curve.tp, curve.fp, curve.fn, beta)
    best = int(np.flatnonzero(scores == scores.max())[-1])
    return float(curve.thresholds[best])


def precision_at_recall_threshold(
    y_true: ArrayLike, y_prob: ArrayLike, min_recall: float = 0.80, n_grid: int | None = None
) -> float:
    """Threshold with the highest precision subject to recall >= ``min_recall``.

    Ties in precision are broken towards the higher threshold (fewer alerts).

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        min_recall: Minimum recall required.
        n_grid: Optional quantile-grid size instead of exhaustive search.

    Returns:
        The selected threshold.

    Raises:
        ValueError: If no candidate threshold reaches ``min_recall``.
    """
    curve = threshold_curve(y_true, y_prob, candidate_thresholds(y_prob, n_grid))
    recall = safe_divide(curve.tp, curve.tp + curve.fn)
    feasible = recall >= min_recall
    if not feasible.any():
        raise ValueError(f"No threshold achieves recall >= {min_recall}")
    precision = np.where(feasible, safe_divide(curve.tp, curve.tp + curve.fp), -1.0)
    best = int(np.flatnonzero(precision == precision.max())[-1])
    return float(curve.thresholds[best])


def cost_threshold(fn_cost: float = DEFAULT_FN_COST, fp_cost: float = DEFAULT_FP_COST) -> float:
    """Bayes-optimal threshold for calibrated probabilities: ``fp / (fp + fn)``.

    Args:
        fn_cost: Cost of a false negative.
        fp_cost: Cost of a false positive.

    Returns:
        The threshold (1/6 for the default 5:1 cost ratio).
    """
    if fn_cost <= 0 or fp_cost <= 0:
        raise ValueError("Costs must be positive")
    return float(fp_cost / (fp_cost + fn_cost))


def min_cost_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
    n_grid: int | None = None,
) -> float:
    """Threshold minimising empirical expected cost (useful when probabilities are uncalibrated).

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        fn_cost: Cost of a false negative.
        fp_cost: Cost of a false positive.
        n_grid: Optional quantile-grid size instead of exhaustive search.

    Returns:
        The cost-minimising threshold (ties broken towards the higher threshold).
    """
    curve = threshold_curve(y_true, y_prob, candidate_thresholds(y_prob, n_grid))
    cost = fn_cost * curve.fn + fp_cost * curve.fp
    best = int(np.flatnonzero(cost == cost.min())[-1])
    return float(curve.thresholds[best])


def operating_point(
    y_true: ArrayLike, y_prob: ArrayLike, threshold: float, policy: str, beta: float = 2.0
) -> ThresholdResult:
    """Describe the operating point (precision, recall, F-beta) of a threshold.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        threshold: Decision threshold.
        policy: Name of the policy that produced the threshold.
        beta: F-beta weight to report.

    Returns:
        The threshold result.
    """
    curve = threshold_curve(y_true, y_prob, np.array([threshold]))
    tp, fp, fn = curve.tp[0], curve.fp[0], curve.fn[0]
    n = tp + fp + fn + curve.tn[0]
    return ThresholdResult(
        policy=policy,
        threshold=float(threshold),
        precision=float(safe_divide(tp, tp + fp)),
        recall=float(safe_divide(tp, tp + fn)),
        fbeta=float(fbeta_from_counts(tp, fp, fn, beta)),
        beta=beta,
        pred_pos_rate=float((tp + fp) / n),
    )


def choose_threshold(
    y_true: ArrayLike, y_prob: ArrayLike, config: ThresholdConfig | None = None
) -> ThresholdResult:
    """Dispatch to the configured threshold policy and report the resulting operating point.

    Args:
        y_true: Binary labels (CAL split).
        y_prob: Positive-class probabilities (CAL split, after calibration if any).
        config: Policy configuration (default: max F2).

    Returns:
        The chosen threshold with its operating point on the given data.
    """
    cfg = config or ThresholdConfig()
    policy = ThresholdPolicy(cfg.policy)
    if policy is ThresholdPolicy.FIXED:
        threshold = cfg.fixed
    elif policy is ThresholdPolicy.MAX_FBETA:
        threshold = max_fbeta_threshold(y_true, y_prob, cfg.beta, cfg.n_grid)
    elif policy is ThresholdPolicy.PRECISION_AT_RECALL:
        threshold = precision_at_recall_threshold(y_true, y_prob, cfg.min_recall, cfg.n_grid)
    elif policy is ThresholdPolicy.COST_BAYES:
        threshold = cost_threshold(cfg.fn_cost, cfg.fp_cost)
    else:
        threshold = min_cost_threshold(y_true, y_prob, cfg.fn_cost, cfg.fp_cost, cfg.n_grid)
    result = operating_point(y_true, y_prob, threshold, policy.value, cfg.beta)
    logger.debug(f"Threshold policy {policy.value}: t={threshold:.6f}")
    return result
