"""Paired significance tests for comparing two classifiers on the same test rows.

* :func:`paired_bootstrap` / :func:`paired_bootstrap_many` - paired (optionally class-stratified)
  bootstrap of a metric difference with percentile CIs and a two-sided p-value.
  Each replicate is represented by integer multiplicity weights; ranking metrics are evaluated from
  a single pre-sort, so a replicate costs O(n) instead of O(n log n).
* :func:`delong_test` - fast DeLong test for correlated ROC-AUCs (Sun & Xu, 2014).
* :func:`mcnemar_test` - McNemar test on hard predictions (exact binomial or chi-square with
  continuity correction).
* :func:`holm_correction` - Holm-Bonferroni step-down adjusted p-values.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as sps

from src.evaluation._checks import check_binary_inputs, check_binary_predictions
from src.evaluation.metrics import (
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    EPS,
    fbeta_from_counts,
    inner,
    safe_divide,
    sort_scores,
    weighted_average_precision,
    weighted_roc_auc,
)

logger = logging.getLogger(__name__)

PlainMetric = Callable[[NDArray[np.int64], NDArray[np.float64]], float]


# --------------------------------------------------------------------------------------------
# Weighted metric protocol and implementations
# --------------------------------------------------------------------------------------------
@runtime_checkable
class WeightedMetric(Protocol):
    """Metric evaluated under integer bootstrap multiplicity weights.

    ``prepare`` runs once per score vector (e.g. sorting); ``evaluate`` runs once per replicate.
    """

    def prepare(self, y: NDArray[np.int64], p: NDArray[np.float64]) -> Any:
        """Precompute replicate-independent state."""

    def evaluate(self, state: Any, weights: NDArray[np.float64]) -> float:
        """Evaluate the metric for one replicate given per-sample weights."""


class RocAucMetric:
    """Weighted ROC-AUC from a single pre-sort."""

    def prepare(self, y: NDArray[np.int64], p: NDArray[np.float64]) -> Any:
        """Sort scores once.

        Args:
            y: Labels.
            p: Scores.

        Returns:
            Sorted-score state.
        """
        return sort_scores(y, p)

    def evaluate(self, state: Any, weights: NDArray[np.float64]) -> float:
        """Weighted ROC-AUC for one replicate.

        Args:
            state: Output of :meth:`prepare`.
            weights: Per-sample weights in original order.

        Returns:
            ROC-AUC.
        """
        return weighted_roc_auc(state, weights[state.order])


class AveragePrecisionMetric:
    """Weighted average precision (PR-AUC) from a single pre-sort."""

    def prepare(self, y: NDArray[np.int64], p: NDArray[np.float64]) -> Any:
        """Sort scores once.

        Args:
            y: Labels.
            p: Scores.

        Returns:
            Sorted-score state.
        """
        return sort_scores(y, p)

    def evaluate(self, state: Any, weights: NDArray[np.float64]) -> float:
        """Weighted average precision for one replicate.

        Args:
            state: Output of :meth:`prepare`.
            weights: Per-sample weights in original order.

        Returns:
            Average precision.
        """
        return weighted_average_precision(state, weights[state.order])


class BrierMetric:
    """Weighted Brier score."""

    def prepare(self, y: NDArray[np.int64], p: NDArray[np.float64]) -> Any:
        """Precompute squared errors.

        Args:
            y: Labels.
            p: Probabilities.

        Returns:
            Per-sample squared errors.
        """
        return (p - y) ** 2

    def evaluate(self, state: Any, weights: NDArray[np.float64]) -> float:
        """Weighted mean squared error.

        Args:
            state: Per-sample squared errors.
            weights: Per-sample weights.

        Returns:
            Brier score.
        """
        return float(inner(weights, state) / weights.sum())


class LogLossMetric:
    """Weighted binary log-loss."""

    def prepare(self, y: NDArray[np.int64], p: NDArray[np.float64]) -> Any:
        """Precompute per-sample negative log-likelihoods.

        Args:
            y: Labels.
            p: Probabilities.

        Returns:
            Per-sample losses.
        """
        pc = np.clip(p, EPS, 1.0 - EPS)
        return -(y * np.log(pc) + (1 - y) * np.log1p(-pc))

    def evaluate(self, state: Any, weights: NDArray[np.float64]) -> float:
        """Weighted mean log-loss.

        Args:
            state: Per-sample losses.
            weights: Per-sample weights.

        Returns:
            Log-loss.
        """
        return float(inner(weights, state) / weights.sum())


@dataclass(frozen=True)
class ThresholdMetric:
    """Weighted threshold-dependent metric (predict positive when ``p >= threshold``).

    Attributes:
        name: One of ``precision``, ``recall``, ``specificity``, ``f1``, ``f2``, ``fbeta``,
            ``balanced_accuracy``, ``accuracy``, ``pred_pos_rate``, ``expected_cost``.
        threshold: Decision threshold.
        beta: Beta for ``fbeta``.
        fn_cost: False-negative cost for ``expected_cost``.
        fp_cost: False-positive cost for ``expected_cost``.
    """

    name: str
    threshold: float
    beta: float = 2.0
    fn_cost: float = DEFAULT_FN_COST
    fp_cost: float = DEFAULT_FP_COST

    def prepare(self, y: NDArray[np.int64], p: NDArray[np.float64]) -> Any:
        """Binarise once.

        Args:
            y: Labels.
            p: Probabilities.

        Returns:
            Tuple of float indicator arrays ``(y, pred, y * pred)``.
        """
        yf = y.astype(np.float64)
        pred = (p >= self.threshold).astype(np.float64)
        return yf, pred, yf * pred

    def evaluate(self, state: Any, weights: NDArray[np.float64]) -> float:
        """Weighted metric from weighted confusion counts.

        Args:
            state: Output of :meth:`prepare`.
            weights: Per-sample weights.

        Returns:
            Metric value.
        """
        yf, pred, both = state
        n = weights.sum()
        tp = inner(weights, both)
        pos = inner(weights, yf)
        pred_pos = inner(weights, pred)
        fp, fn = pred_pos - tp, pos - tp
        tn = n - tp - fp - fn
        return self._from_counts(tp, fp, fn, tn)

    def _from_counts(self, tp: float, fp: float, fn: float, tn: float) -> float:
        """Map (weighted) confusion counts to the named metric.

        Args:
            tp: True positives.
            fp: False positives.
            fn: False negatives.
            tn: True negatives.

        Returns:
            Metric value.

        Raises:
            ValueError: If the metric name is unknown.
        """
        n = tp + fp + fn + tn
        recall = float(safe_divide(tp, tp + fn))
        specificity = float(safe_divide(tn, tn + fp))
        values: dict[str, Callable[[], float]] = {
            "precision": lambda: float(safe_divide(tp, tp + fp)),
            "recall": lambda: recall,
            "specificity": lambda: specificity,
            "f1": lambda: float(fbeta_from_counts(tp, fp, fn, 1.0)),
            "f2": lambda: float(fbeta_from_counts(tp, fp, fn, 2.0)),
            "fbeta": lambda: float(fbeta_from_counts(tp, fp, fn, self.beta)),
            "balanced_accuracy": lambda: 0.5 * (recall + specificity),
            "accuracy": lambda: (tp + tn) / n,
            "pred_pos_rate": lambda: (tp + fp) / n,
            "expected_cost": lambda: (self.fn_cost * fn + self.fp_cost * fp) / n,
        }
        if self.name not in values:
            raise ValueError(f"Unknown threshold metric {self.name!r}")
        return values[self.name]()


class CallableMetric:
    """Adapter running an arbitrary ``metric(y, p) -> float`` on materialised resamples (slow path).

    Use this for metrics without a weighted implementation (e.g. ECE). Rows are repeated according
    to their bootstrap multiplicity, so the metric must be invariant to row order.
    """

    def __init__(self, fn: PlainMetric) -> None:
        """Initialise.

        Args:
            fn: Metric function taking ``(y_true, y_prob)``.
        """
        self.fn = fn

    def prepare(self, y: NDArray[np.int64], p: NDArray[np.float64]) -> Any:
        """Store the arrays.

        Args:
            y: Labels.
            p: Probabilities.

        Returns:
            Tuple ``(y, p, row_index)``.
        """
        return y, p, np.arange(y.size)

    def evaluate(self, state: Any, weights: NDArray[np.float64]) -> float:
        """Materialise the resample and call the wrapped function.

        Args:
            state: Output of :meth:`prepare`.
            weights: Integer multiplicity weights.

        Returns:
            Metric value.
        """
        y, p, rows = state
        idx = np.repeat(rows, weights.astype(np.int64))
        return float(self.fn(y[idx], p[idx]))


FAST_METRICS: dict[str, Callable[[], WeightedMetric]] = {
    "roc_auc": RocAucMetric,
    "pr_auc": AveragePrecisionMetric,
    "average_precision": AveragePrecisionMetric,
    "brier": BrierMetric,
    "log_loss": LogLossMetric,
}

MetricSpec = str | WeightedMetric | PlainMetric


def resolve_metric(spec: MetricSpec) -> WeightedMetric:
    """Turn a metric name, weighted metric, or plain callable into a :class:`WeightedMetric`.

    Args:
        spec: Name in :data:`FAST_METRICS`, a :class:`WeightedMetric` (e.g.
            ``ThresholdMetric("f2", 0.31)``), or a plain ``fn(y, p) -> float``.

    Returns:
        A weighted metric.

    Raises:
        ValueError: If a string name is unknown.
    """
    if isinstance(spec, str):
        if spec not in FAST_METRICS:
            raise ValueError(f"Unknown metric {spec!r}; choose from {sorted(FAST_METRICS)}")
        return FAST_METRICS[spec]()
    if isinstance(spec, WeightedMetric):
        return spec
    return CallableMetric(spec)


# --------------------------------------------------------------------------------------------
# Paired bootstrap
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapResult:
    """Paired-bootstrap comparison of model A vs model B (``delta = A - B``)."""

    metric_a: float
    metric_b: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    n_boot: int
    n_valid: int
    alpha: float
    deltas: NDArray[np.float64] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return scalar fields as a dict (bootstrap samples excluded)."""
        out = asdict(self)
        out.pop("deltas")
        return out


def bootstrap_weights(
    y: NDArray[np.int64], n_boot: int, seed: int, stratified: bool = True
) -> Iterator[NDArray[np.float64]]:
    """Yield bootstrap multiplicity weights, one vector per replicate.

    With ``stratified=True`` positives and negatives are resampled separately so each replicate
    keeps the observed class counts (stabilises PR-AUC at low prevalence).

    Args:
        y: Binary labels.
        n_boot: Number of replicates.
        seed: RNG seed.
        stratified: Whether to resample within each class.

    Yields:
        Float array of length ``n`` with non-negative integer counts summing to ``n``.
    """
    rng = np.random.default_rng(seed)
    n = y.size
    strata = [np.flatnonzero(y == 1), np.flatnonzero(y == 0)] if stratified else [np.arange(n)]
    strata = [s for s in strata if s.size > 0]
    for _ in range(n_boot):
        weights = np.zeros(n, dtype=np.float64)
        for members in strata:
            draws = rng.integers(0, members.size, size=members.size)
            weights[members] = np.bincount(draws, minlength=members.size)
        yield weights


def bootstrap_p_value(deltas: NDArray[np.float64]) -> float:
    """Two-sided proportion-based p-value for H0: delta = 0.

    ``p = min(1, 2 * min(#{d <= 0} + 1, #{d >= 0} + 1) / (B + 1))`` - the +1 terms keep the
    p-value strictly positive given a finite number of replicates.

    Args:
        deltas: Bootstrap replicates of the difference (NaNs ignored).

    Returns:
        The p-value.
    """
    d = deltas[np.isfinite(deltas)]
    if d.size == 0:
        return float("nan")
    k_le = int(np.count_nonzero(d <= 0))
    k_ge = int(np.count_nonzero(d >= 0))
    return float(min(1.0, 2.0 * (min(k_le, k_ge) + 1) / (d.size + 1)))


def _percentile_ci(samples: NDArray[np.float64], alpha: float) -> tuple[float, float]:
    """Percentile confidence interval ignoring NaNs.

    Args:
        samples: Bootstrap replicates.
        alpha: Significance level (0.05 -> 95% CI).

    Returns:
        Tuple ``(low, high)``.
    """
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def paired_bootstrap_many(
    y_true: ArrayLike,
    p_a: ArrayLike,
    p_b: ArrayLike,
    metrics: Mapping[str, MetricSpec],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    stratified: bool = True,
    metrics_b: Mapping[str, MetricSpec] | None = None,
) -> dict[str, BootstrapResult]:
    """Paired bootstrap for several metrics sharing the same resamples.

    Args:
        y_true: Binary labels.
        p_a: Model A probabilities.
        p_b: Model B probabilities.
        metrics: Mapping of result name to metric spec (see :func:`resolve_metric`) for model A
            (and model B unless ``metrics_b`` is given).
        n_boot: Number of bootstrap replicates.
        seed: RNG seed (same resamples for A and B, and across metrics).
        alpha: Significance level for the percentile CIs.
        stratified: Resample within classes.
        metrics_b: Optional per-name metric specs for model B, e.g. a ``ThresholdMetric`` at
            model B's own threshold. Keys must match ``metrics``.

    Returns:
        Mapping of metric name to bootstrap result.
    """
    y, pa = check_binary_inputs(y_true, p_a)
    _, pb = check_binary_inputs(y, p_b)
    specs_b = metrics_b if metrics_b is not None else metrics
    if set(specs_b) != set(metrics):
        raise ValueError("metrics_b must have the same keys as metrics")
    resolved = {k: (resolve_metric(metrics[k]), resolve_metric(specs_b[k])) for k in metrics}
    states = {k: (ma.prepare(y, pa), mb.prepare(y, pb)) for k, (ma, mb) in resolved.items()}
    ones = np.ones(y.size, dtype=np.float64)
    point = {k: (ma.evaluate(states[k][0], ones), mb.evaluate(states[k][1], ones))
             for k, (ma, mb) in resolved.items()}
    samples = {k: np.empty((n_boot, 2), dtype=np.float64) for k in metrics}
    for b, weights in enumerate(bootstrap_weights(y, n_boot, seed, stratified)):
        for k, (ma, mb) in resolved.items():
            samples[k][b, 0] = ma.evaluate(states[k][0], weights)
            samples[k][b, 1] = mb.evaluate(states[k][1], weights)
    return {k: _summarise(point[k], samples[k], alpha) for k in metrics}


def _summarise(
    point: tuple[float, float], samples: NDArray[np.float64], alpha: float
) -> BootstrapResult:
    """Summarise bootstrap replicates of (metric_a, metric_b) into a result.

    Args:
        point: Full-sample ``(metric_a, metric_b)``.
        samples: Array of shape (B, 2) of replicate metric values.
        alpha: Significance level.

    Returns:
        The bootstrap result.
    """
    deltas = samples[:, 0] - samples[:, 1]
    ci_low, ci_high = _percentile_ci(deltas, alpha)
    return BootstrapResult(
        metric_a=point[0],
        metric_b=point[1],
        delta=point[0] - point[1],
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=bootstrap_p_value(deltas),
        ci_a=_percentile_ci(samples[:, 0], alpha),
        ci_b=_percentile_ci(samples[:, 1], alpha),
        n_boot=int(samples.shape[0]),
        n_valid=int(np.count_nonzero(np.isfinite(deltas))),
        alpha=alpha,
        deltas=deltas,
    )


def paired_bootstrap(
    y_true: ArrayLike,
    p_a: ArrayLike,
    p_b: ArrayLike,
    metric_fn: MetricSpec = "pr_auc",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    stratified: bool = True,
    metric_fn_b: MetricSpec | None = None,
) -> BootstrapResult:
    """Paired bootstrap of ``metric(A) - metric(B)`` with percentile CI and two-sided p-value.

    Args:
        y_true: Binary labels.
        p_a: Model A probabilities.
        p_b: Model B probabilities.
        metric_fn: Metric name (``"pr_auc"``, ``"roc_auc"``, ``"brier"``, ``"log_loss"``), a
            :class:`WeightedMetric` such as ``ThresholdMetric("f2", t)``, or any plain
            ``fn(y, p) -> float`` (slower; e.g. a thresholded or ECE closure).
        n_boot: Number of replicates.
        seed: RNG seed.
        alpha: Significance level (0.05 -> 95% CI).
        stratified: Resample within classes.
        metric_fn_b: Optional different metric for model B (e.g. its own threshold).

    Returns:
        The bootstrap result.
    """
    metrics_b = None if metric_fn_b is None else {"m": metric_fn_b}
    return paired_bootstrap_many(
        y_true, p_a, p_b, {"m": metric_fn}, n_boot, seed, alpha, stratified, metrics_b
    )["m"]


# --------------------------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DeLongResult:
    """DeLong test for two correlated ROC-AUCs (``delta = auc_a - auc_b``)."""

    auc_a: float
    auc_b: float
    delta: float
    se: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float]:
        """Return the result as a dict."""
        return asdict(self)


def _delong_components(
    y: NDArray[np.int64], scores: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """AUCs and structural components (V10, V01) via midranks (Sun & Xu, 2014).

    Args:
        y: Binary labels, shape (n,).
        scores: Score matrix, shape (k, n), one row per classifier.

    Returns:
        Tuple ``(aucs (k,), v10 (k, m), v01 (k, n_neg))``.
    """
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n_neg = pos.shape[1], neg.shape[1]
    tx = sps.rankdata(pos, axis=1)
    ty = sps.rankdata(neg, axis=1)
    tz = sps.rankdata(np.concatenate([pos, neg], axis=1), axis=1)
    aucs = tz[:, :m].sum(axis=1) / (m * n_neg) - (m + 1.0) / (2.0 * n_neg)
    v10 = (tz[:, :m] - tx) / n_neg
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v10, v01


def delong_test(
    y_true: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, alpha: float = 0.05
) -> DeLongResult:
    """Fast DeLong test for the difference of two correlated ROC-AUCs.

    Args:
        y_true: Binary labels.
        p_a: Model A scores.
        p_b: Model B scores.
        alpha: Significance level for the normal-approximation CI of the difference.

    Returns:
        AUCs, difference, standard error, z statistic, two-sided p-value and CI.
        Identical score vectors give ``se = 0``, ``z = 0`` and ``p = 1``.
    """
    y, pa = check_binary_inputs(y_true, p_a)
    _, pb = check_binary_inputs(y, p_b)
    if y.min() == y.max():
        raise ValueError("DeLong test requires both classes in y_true")
    aucs, v10, v01 = _delong_components(y, np.vstack([pa, pb]))
    cov = np.cov(v10) / v10.shape[1] + np.cov(v01) / v01.shape[1]
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    delta = float(aucs[0] - aucs[1])
    se = float(np.sqrt(max(var, 0.0)))
    if se <= 1e-15:
        # Degenerate variance: identical AUCs -> no evidence; different AUCs -> exact separation.
        z = 0.0 if abs(delta) <= 1e-15 else float(np.sign(delta) * np.inf)
        p_value = 1.0 if abs(delta) <= 1e-15 else 0.0
    else:
        z = delta / se
        p_value = float(2.0 * sps.norm.sf(abs(z)))
    z_crit = float(sps.norm.ppf(1.0 - alpha / 2.0))
    return DeLongResult(
        auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta=delta, se=se, z=float(z),
        p_value=p_value, ci_low=delta - z_crit * se, ci_high=delta + z_crit * se,
    )


# --------------------------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class McNemarResult:
    """McNemar test on discordant pairs.

    Attributes:
        n01: A correct, B wrong.
        n10: A wrong, B correct.
        statistic: Chi-square statistic (continuity-corrected) or min(n01, n10) for exact.
        p_value: Two-sided p-value.
        method: ``"exact"`` or ``"chi2"``.
    """

    n01: int
    n10: int
    statistic: float
    p_value: float
    method: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Return the result as a dict."""
        return asdict(self)


def mcnemar_from_counts(n01: int, n10: int, exact_below: int = 25) -> McNemarResult:
    """McNemar test from the two discordant counts.

    Args:
        n01: Pairs where A is correct and B is wrong.
        n10: Pairs where A is wrong and B is correct.
        exact_below: Use the exact binomial test when ``n01 + n10 < exact_below``.

    Returns:
        The test result.
    """
    total = n01 + n10
    if total == 0:
        return McNemarResult(n01=n01, n10=n10, statistic=0.0, p_value=1.0, method="exact")
    if total < exact_below:
        k = min(n01, n10)
        p_value = float(min(1.0, 2.0 * sps.binom.cdf(k, total, 0.5)))
        return McNemarResult(n01=n01, n10=n10, statistic=float(k), p_value=p_value,
                             method="exact")
    statistic = (abs(n01 - n10) - 1.0) ** 2 / total
    return McNemarResult(n01=n01, n10=n10, statistic=float(statistic),
                         p_value=float(sps.chi2.sf(statistic, df=1)), method="chi2")


def mcnemar_test(
    y_true: ArrayLike, pred_a: ArrayLike, pred_b: ArrayLike, exact_below: int = 25
) -> McNemarResult:
    """McNemar test comparing the error patterns of two hard classifiers on the same rows.

    Args:
        y_true: Binary labels.
        pred_a: Model A 0/1 predictions.
        pred_b: Model B 0/1 predictions.
        exact_below: Use the exact binomial test when the discordant total is below this.

    Returns:
        The test result.
    """
    y = check_binary_predictions(y_true, np.asarray(y_true).size)
    correct_a = check_binary_predictions(pred_a, y.size) == y
    correct_b = check_binary_predictions(pred_b, y.size) == y
    n01 = int(np.count_nonzero(correct_a & ~correct_b))
    n10 = int(np.count_nonzero(~correct_a & correct_b))
    return mcnemar_from_counts(n01, n10, exact_below)


# --------------------------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------------------------
def holm_correction(p_values: ArrayLike) -> NDArray[np.float64]:
    """Holm-Bonferroni step-down adjusted p-values (returned in the input order).

    Args:
        p_values: Raw p-values.

    Returns:
        Adjusted p-values, monotone in the raw ordering and capped at 1.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    scaled = (m - np.arange(m)) * p[order]
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted
