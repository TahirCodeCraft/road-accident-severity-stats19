"""Binary and multiclass evaluation metrics for severity classification.

All binary functions take ``(y_true, y_prob)`` where ``y_prob`` is the predicted probability of the
positive (severe) class. Ranking metrics have fast sort-based implementations that also accept
per-sample multiplicity weights so that bootstrap replicates can be evaluated in O(n) after a single
O(n log n) sort (see :mod:`src.evaluation.stats`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.evaluation._checks import check_binary_inputs

logger = logging.getLogger(__name__)

BinStrategy = Literal["quantile", "uniform"]
EPS = 1e-15
DEFAULT_FN_COST = 5.0
DEFAULT_FP_COST = 1.0
DEFAULT_N_BINS = 15


# --------------------------------------------------------------------------------------------
# Result structures
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfusionCounts:
    """Binary confusion-matrix counts."""

    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class BinaryMetrics:
    """Full set of threshold-dependent and threshold-free binary metrics."""

    n: int
    n_pos: int
    prevalence: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    f1: float
    f2: float
    balanced_accuracy: float
    pred_pos_rate: float
    pr_auc: float
    roc_auc: float
    brier: float
    log_loss: float
    ece: float
    expected_cost: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the metrics as a plain dict (e.g. for wandb or JSON)."""
        return asdict(self)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin reliability-diagram data (empty bins removed)."""

    mean_pred: NDArray[np.float64]
    frac_pos: NDArray[np.float64]
    count: NDArray[np.int64]
    bin_edges: NDArray[np.float64]
    strategy: str


@dataclass(frozen=True)
class MulticlassMetrics:
    """Multiclass summary metrics (used for the 3-class severity sensitivity analysis)."""

    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    labels: list[int]
    per_class_recall: dict[int, float]
    per_class_precision: dict[int, float]
    per_class_f1: dict[int, float]
    confusion: NDArray[np.int64]

    def to_dict(self) -> dict[str, float]:
        """Flatten scalar and per-class metrics into a dict (confusion matrix excluded)."""
        out: dict[str, float] = {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
        }
        for label in self.labels:
            out[f"recall_{label}"] = self.per_class_recall[label]
            out[f"precision_{label}"] = self.per_class_precision[label]
            out[f"f1_{label}"] = self.per_class_f1[label]
        return out


@dataclass(frozen=True)
class SortedScores:
    """Scores sorted descending with tie groups, reusable across weighted evaluations."""

    order: NDArray[np.int64]
    y_sorted: NDArray[np.float64]
    group_id: NDArray[np.int64]
    n_groups: int


# --------------------------------------------------------------------------------------------
# Threshold-dependent primitives
# --------------------------------------------------------------------------------------------
def safe_divide(num: ArrayLike, den: ArrayLike) -> NDArray[np.float64] | float:
    """Element-wise division returning 0 where the denominator is 0.

    Args:
        num: Numerator(s).
        den: Denominator(s).

    Returns:
        Quotient with zero-division mapped to 0 (scalar if inputs are scalar).
    """
    num_arr = np.asarray(num, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    out = np.divide(num_arr, den_arr, out=np.zeros(np.broadcast(num_arr, den_arr).shape),
                    where=den_arr != 0)
    return float(out) if out.ndim == 0 else out


def fbeta_from_counts(
    tp: ArrayLike, fp: ArrayLike, fn: ArrayLike, beta: float = 1.0
) -> NDArray[np.float64] | float:
    """F-beta score from confusion counts (vectorised; 0 when undefined).

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        beta: Recall weight (beta=2 weights recall 4x as much as precision in the harmonic mean).

    Returns:
        F-beta score(s).
    """
    b2 = beta * beta
    tp_arr = np.asarray(tp, dtype=np.float64)
    return safe_divide((1.0 + b2) * tp_arr, (1.0 + b2) * tp_arr + b2 * np.asarray(fn) +
                       np.asarray(fp))


def confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> ConfusionCounts:
    """Compute binary confusion counts.

    Args:
        y_true: Binary labels.
        y_pred: Binary predictions.

    Returns:
        The confusion counts.
    """
    y = np.asarray(y_true).astype(bool).ravel()
    pred = np.asarray(y_pred).astype(bool).ravel()
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    return ConfusionCounts(tp=tp, fp=fp, tn=int(y.size - tp - fp - fn), fn=fn)


def expected_cost(
    counts: ConfusionCounts,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> float:
    """Average misclassification cost per sample.

    Args:
        counts: Confusion counts.
        fn_cost: Cost of a missed severe collision.
        fp_cost: Cost of a false alarm.

    Returns:
        ``(fn_cost * FN + fp_cost * FP) / n``.
    """
    n = counts.tp + counts.fp + counts.tn + counts.fn
    return float((fn_cost * counts.fn + fp_cost * counts.fp) / n)


# --------------------------------------------------------------------------------------------
# Probabilistic / calibration metrics
# --------------------------------------------------------------------------------------------
def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Brier score (mean squared error of probabilities).

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.

    Returns:
        The Brier score.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def log_loss_binary(y_true: ArrayLike, y_prob: ArrayLike, eps: float = EPS) -> float:
    """Binary cross-entropy with probability clipping.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        eps: Clipping constant.

    Returns:
        Mean negative log-likelihood.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def calibration_bin_ids(
    y_prob: ArrayLike, n_bins: int = DEFAULT_N_BINS, strategy: BinStrategy = "quantile"
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Assign each probability to a calibration bin.

    ``"quantile"`` gives equal-mass bins (edges at empirical quantiles; duplicate edges caused by
    ties are merged), ``"uniform"`` gives equal-width bins on [0, 1].

    Args:
        y_prob: Probabilities.
        n_bins: Requested number of bins.
        strategy: ``"quantile"`` or ``"uniform"``.

    Returns:
        Tuple ``(bin_ids, edges)`` where ``edges`` includes the outer bounds.

    Raises:
        ValueError: If ``strategy`` is unknown or ``n_bins < 1``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        interior = edges[1:-1]
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        interior = edges[1:-1]
    else:
        raise ValueError(f"Unknown binning strategy {strategy!r}")
    return np.searchsorted(interior, p, side="right").astype(np.int64), edges


def reliability_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = DEFAULT_N_BINS,
    strategy: BinStrategy = "quantile",
) -> ReliabilityCurve:
    """Compute reliability-diagram data: mean prediction vs observed rate per bin.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        n_bins: Number of bins.
        strategy: ``"quantile"`` (equal-mass) or ``"uniform"`` (equal-width).

    Returns:
        Non-empty bins' mean predicted probability, observed positive fraction and counts.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    ids, edges = calibration_bin_ids(p, n_bins, strategy)
    n_slots = max(len(edges) - 1, int(ids.max()) + 1)
    count = np.bincount(ids, minlength=n_slots)
    sum_p = np.bincount(ids, weights=p, minlength=n_slots)
    sum_y = np.bincount(ids, weights=y, minlength=n_slots)
    keep = count > 0
    return ReliabilityCurve(
        mean_pred=sum_p[keep] / count[keep],
        frac_pos=sum_y[keep] / count[keep],
        count=count[keep].astype(np.int64),
        bin_edges=edges,
        strategy=strategy,
    )


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = DEFAULT_N_BINS,
    strategy: BinStrategy = "quantile",
) -> float:
    """Expected calibration error: count-weighted mean |observed rate - mean prediction|.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.
        n_bins: Number of bins (15 by project convention).
        strategy: ``"quantile"`` (equal-mass, default) or ``"uniform"`` (equal-width).

    Returns:
        The ECE in [0, 1].
    """
    curve = reliability_curve(y_true, y_prob, n_bins, strategy)
    weights = curve.count / curve.count.sum()
    return float(np.sum(weights * np.abs(curve.frac_pos - curve.mean_pred)))


# --------------------------------------------------------------------------------------------
# Fast (optionally weighted) ranking metrics
# --------------------------------------------------------------------------------------------
def sort_scores(y_true: ArrayLike, y_prob: ArrayLike) -> SortedScores:
    """Sort scores in descending order and label tie groups (done once per score vector).

    Args:
        y_true: Binary labels.
        y_prob: Scores or probabilities (higher = more likely positive).

    Returns:
        The sorted-score structure consumed by the weighted ranking metrics.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    new_group = np.empty(p_sorted.size, dtype=bool)
    new_group[0] = True
    np.not_equal(p_sorted[1:], p_sorted[:-1], out=new_group[1:])
    group_id = np.cumsum(new_group) - 1
    return SortedScores(order=order, y_sorted=y[order], group_id=group_id,
                        n_groups=int(group_id[-1]) + 1)


def inner(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Single-threaded inner product of two 1-D arrays.

    ``np.dot`` dispatches to multithreaded BLAS, which spin-waits badly when called thousands of
    times on a loaded machine (e.g. inside a bootstrap loop); ``einsum`` stays on one core.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The inner product.
    """
    return float(np.einsum("i,i->", a, b))


def _group_weights(
    scores: SortedScores, w_sorted: NDArray[np.float64] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Positive weight and total weight in each tie group (descending score order).

    When there are no ties each group is a single sample and the aggregation is skipped.

    Args:
        scores: Output of :func:`sort_scores`.
        w_sorted: Sample weights aligned with the sorted order, or ``None`` for unit weights.

    Returns:
        Tuple ``(pos_weight_per_group, total_weight_per_group)``.
    """
    w = np.ones_like(scores.y_sorted) if w_sorted is None else w_sorted
    wy = w * scores.y_sorted
    if scores.n_groups == w.size:
        return wy, w
    pos = np.bincount(scores.group_id, weights=wy, minlength=scores.n_groups)
    total = np.bincount(scores.group_id, weights=w, minlength=scores.n_groups)
    return pos, total


def weighted_average_precision(
    scores: SortedScores, w_sorted: NDArray[np.float64] | None = None
) -> float:
    """Average precision (sklearn definition, step-wise, ties grouped) with sample weights.

    Args:
        scores: Output of :func:`sort_scores`.
        w_sorted: Sample weights aligned with ``scores.order`` (``None`` = unweighted).

    Returns:
        Average precision, or NaN if there is no positive weight.
    """
    pos, total = _group_weights(scores, w_sorted)
    tps = np.cumsum(pos)
    total_pos = tps[-1]
    if total_pos <= 0:
        return float("nan")
    precision = safe_divide(tps, np.cumsum(total))
    return float(inner(pos, precision) / total_pos)


def weighted_roc_auc(scores: SortedScores, w_sorted: NDArray[np.float64] | None = None) -> float:
    """ROC-AUC (Mann-Whitney with ties counted 1/2) with sample weights.

    Args:
        scores: Output of :func:`sort_scores`.
        w_sorted: Sample weights aligned with ``scores.order`` (``None`` = unweighted).

    Returns:
        ROC-AUC, or NaN if either class has zero weight.
    """
    pos, total = _group_weights(scores, w_sorted)
    neg = total - pos
    neg_through = np.cumsum(neg)
    total_pos = pos.sum()
    total_neg = neg_through[-1]
    if total_pos <= 0 or total_neg <= 0:
        return float("nan")
    # Negatives strictly below group g = total_neg - neg_through[g]; ties count one half.
    numerator = total_neg * total_pos - inner(pos, neg_through) + 0.5 * inner(pos, neg)
    return float(numerator / (total_pos * total_neg))


def average_precision_fast(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Sort-based average precision equal to :func:`sklearn.metrics.average_precision_score`.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class scores.

    Returns:
        Average precision.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    return weighted_average_precision(sort_scores(y, p))


def roc_auc_fast(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Sort-based ROC-AUC equal to :func:`sklearn.metrics.roc_auc_score`.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class scores.

    Returns:
        ROC-AUC.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    return weighted_roc_auc(sort_scores(y, p))


# --------------------------------------------------------------------------------------------
# Aggregate reports
# --------------------------------------------------------------------------------------------
def _ranking_metrics(y: NDArray[np.int64], p: NDArray[np.float64]) -> tuple[float, float]:
    """PR-AUC and ROC-AUC via sklearn, NaN (with a warning) if only one class is present.

    Args:
        y: Validated binary labels.
        p: Validated probabilities.

    Returns:
        Tuple ``(pr_auc, roc_auc)``.
    """
    if y.min() == y.max():
        logger.warning("Only one class present in y_true; PR-AUC/ROC-AUC set to NaN")
        return float("nan"), float("nan")
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def binary_metrics(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    threshold: float = 0.5,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
    n_bins: int = DEFAULT_N_BINS,
    ece_strategy: BinStrategy = "quantile",
) -> BinaryMetrics:
    """Compute the full binary evaluation report at a decision threshold.

    A sample is predicted positive when ``y_prob >= threshold``.

    Args:
        y_true: Binary labels (1 = severe).
        y_prob: Positive-class probabilities.
        threshold: Decision threshold.
        fn_cost: Cost of a false negative for the expected-cost metric.
        fp_cost: Cost of a false positive for the expected-cost metric.
        n_bins: Number of ECE bins.
        ece_strategy: ``"quantile"`` (equal-mass) or ``"uniform"`` ECE binning.

    Returns:
        The metrics dataclass.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    counts = confusion_counts(y, p >= threshold)
    n = y.size
    precision = float(safe_divide(counts.tp, counts.tp + counts.fp))
    recall = float(safe_divide(counts.tp, counts.tp + counts.fn))
    specificity = float(safe_divide(counts.tn, counts.tn + counts.fp))
    pr_auc, roc_auc = _ranking_metrics(y, p)
    return BinaryMetrics(
        n=n,
        n_pos=int(y.sum()),
        prevalence=float(y.mean()),
        threshold=float(threshold),
        tp=counts.tp,
        fp=counts.fp,
        tn=counts.tn,
        fn=counts.fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=float(fbeta_from_counts(counts.tp, counts.fp, counts.fn, beta=1.0)),
        f2=float(fbeta_from_counts(counts.tp, counts.fp, counts.fn, beta=2.0)),
        balanced_accuracy=0.5 * (recall + specificity),
        pred_pos_rate=float((counts.tp + counts.fp) / n),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier=brier_score(y, p),
        log_loss=log_loss_binary(y, p),
        ece=expected_calibration_error(y, p, n_bins, ece_strategy),
        expected_cost=expected_cost(counts, fn_cost, fp_cost),
    )


def multiclass_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, labels: list[int] | None = None
) -> MulticlassMetrics:
    """Macro-F1, accuracy, balanced accuracy and per-class precision/recall/F1.

    Args:
        y_true: Integer class labels.
        y_pred: Predicted class labels (e.g. ``argmax`` of class probabilities).
        labels: Class labels to report; defaults to the sorted union of observed labels.

    Returns:
        The multiclass metrics dataclass.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true and y_pred lengths differ: {yt.shape[0]} vs {yp.shape[0]}")
    label_list = [int(v) for v in (labels if labels is not None else np.union1d(yt, yp))]
    precision, recall, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=label_list, zero_division=0
    )
    return MulticlassMetrics(
        accuracy=float(np.mean(yt == yp)),
        macro_f1=float(np.mean(f1)),
        balanced_accuracy=float(np.mean(recall)),
        labels=label_list,
        per_class_recall={lab: float(v) for lab, v in zip(label_list, recall, strict=True)},
        per_class_precision={lab: float(v) for lab, v in zip(label_list, precision, strict=True)},
        per_class_f1={lab: float(v) for lab, v in zip(label_list, f1, strict=True)},
        confusion=confusion_matrix(yt, yp, labels=label_list).astype(np.int64),
    )
