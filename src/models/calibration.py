"""Post-hoc probability calibrators fitted on the held-out CAL split.

All calibrators share the interface ``fit(y_true, y_prob) -> self`` and ``predict(y_prob)``, map
probabilities to probabilities, and can be persisted with joblib. ``cross_fit_calibrated`` produces
out-of-fold calibrated probabilities so that validation metrics on CAL are not optimistically biased
by fitting and evaluating the calibrator on the same rows.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.evaluation._checks import check_binary_inputs

logger = logging.getLogger(__name__)

PROB_CLIP = 1e-6


class CalibrationMethod(str, Enum):
    """Available calibration methods."""

    IDENTITY = "identity"
    PLATT = "platt"
    ISOTONIC = "isotonic"


def clipped_logit(y_prob: ArrayLike, eps: float = PROB_CLIP) -> NDArray[np.float64]:
    """Logit of probabilities clipped to ``[eps, 1 - eps]``.

    Args:
        y_prob: Probabilities.
        eps: Clipping constant.

    Returns:
        Log-odds array.
    """
    p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


class Calibrator(ABC):
    """Common interface for probability calibrators."""

    method: CalibrationMethod

    @abstractmethod
    def fit(self, y_true: ArrayLike, y_prob: ArrayLike) -> Calibrator:
        """Fit the calibration map.

        Args:
            y_true: Binary labels.
            y_prob: Uncalibrated positive-class probabilities.

        Returns:
            The fitted calibrator (``self``).
        """

    @abstractmethod
    def predict(self, y_prob: ArrayLike) -> NDArray[np.float64]:
        """Map uncalibrated probabilities to calibrated ones.

        Args:
            y_prob: Uncalibrated positive-class probabilities.

        Returns:
            Calibrated probabilities in [0, 1].
        """

    def save(self, path: Path) -> None:
        """Persist the calibrator with joblib.

        Args:
            path: Destination file; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved {self.method.value} calibrator to {path}")


class IdentityCalibrator(Calibrator):
    """No-op calibrator (returns the input probabilities)."""

    method = CalibrationMethod.IDENTITY

    def fit(self, y_true: ArrayLike, y_prob: ArrayLike) -> IdentityCalibrator:
        """No-op fit (inputs validated only).

        Args:
            y_true: Binary labels.
            y_prob: Probabilities.

        Returns:
            ``self``.
        """
        check_binary_inputs(y_true, y_prob)
        return self

    def predict(self, y_prob: ArrayLike) -> NDArray[np.float64]:
        """Return the probabilities unchanged (as float64, clipped to [0, 1]).

        Args:
            y_prob: Probabilities.

        Returns:
            The same probabilities.
        """
        return np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)


class PlattCalibrator(Calibrator):
    """Platt scaling: unpenalised logistic regression on the logit of the clipped probability."""

    method = CalibrationMethod.PLATT

    def __init__(self, eps: float = PROB_CLIP) -> None:
        """Initialise.

        Args:
            eps: Probability clipping constant applied before the logit.
        """
        self.eps = eps
        self.model_: LogisticRegression | None = None

    def fit(self, y_true: ArrayLike, y_prob: ArrayLike) -> PlattCalibrator:
        """Fit slope and intercept on logit(p).

        Args:
            y_true: Binary labels.
            y_prob: Uncalibrated probabilities.

        Returns:
            ``self``.
        """
        y, p = check_binary_inputs(y_true, y_prob)
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        model.fit(clipped_logit(p, self.eps).reshape(-1, 1), y)
        self.model_ = model
        return self

    def predict(self, y_prob: ArrayLike) -> NDArray[np.float64]:
        """Apply the fitted sigmoid map.

        Args:
            y_prob: Uncalibrated probabilities.

        Returns:
            Calibrated probabilities.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model_ is None:
            raise RuntimeError("PlattCalibrator is not fitted")
        z = clipped_logit(y_prob, self.eps).reshape(-1, 1)
        return self.model_.predict_proba(z)[:, 1]

    @property
    def coef(self) -> tuple[float, float]:
        """Fitted ``(slope, intercept)`` on the logit scale."""
        if self.model_ is None:
            raise RuntimeError("PlattCalibrator is not fitted")
        return float(self.model_.coef_[0, 0]), float(self.model_.intercept_[0])


class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibrator (monotone step function, clipped out of bounds)."""

    method = CalibrationMethod.ISOTONIC

    def __init__(self) -> None:
        """Initialise an unfitted isotonic calibrator."""
        self.model_: IsotonicRegression | None = None

    def fit(self, y_true: ArrayLike, y_prob: ArrayLike) -> IsotonicCalibrator:
        """Fit a non-decreasing map from probability to observed frequency.

        Args:
            y_true: Binary labels.
            y_prob: Uncalibrated probabilities.

        Returns:
            ``self``.
        """
        y, p = check_binary_inputs(y_true, y_prob)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        model.fit(p, y.astype(np.float64))
        self.model_ = model
        return self

    def predict(self, y_prob: ArrayLike) -> NDArray[np.float64]:
        """Apply the fitted isotonic map.

        Args:
            y_prob: Uncalibrated probabilities.

        Returns:
            Calibrated probabilities.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model_ is None:
            raise RuntimeError("IsotonicCalibrator is not fitted")
        p = np.asarray(y_prob, dtype=np.float64).ravel()
        return np.asarray(self.model_.predict(p), dtype=np.float64)


_REGISTRY: dict[CalibrationMethod, type[Calibrator]] = {
    CalibrationMethod.IDENTITY: IdentityCalibrator,
    CalibrationMethod.PLATT: PlattCalibrator,
    CalibrationMethod.ISOTONIC: IsotonicCalibrator,
}


def make_calibrator(method: CalibrationMethod | str) -> Calibrator:
    """Instantiate an unfitted calibrator by name.

    Args:
        method: ``"identity"``, ``"platt"`` or ``"isotonic"``.

    Returns:
        A new calibrator instance.
    """
    return _REGISTRY[CalibrationMethod(method)]()


def load_calibrator(path: Path) -> Calibrator:
    """Load a calibrator saved with :meth:`Calibrator.save`.

    Args:
        path: File written by ``save``.

    Returns:
        The calibrator.

    Raises:
        TypeError: If the file does not contain a calibrator.
    """
    obj = joblib.load(Path(path))
    if not isinstance(obj, Calibrator):
        raise TypeError(f"{path} does not contain a Calibrator (got {type(obj).__name__})")
    return obj


def cross_fit_calibrated(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    method: CalibrationMethod | str,
    n_splits: int = 5,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Out-of-fold calibrated probabilities via stratified K-fold cross-fitting.

    Each fold's probabilities are calibrated by a calibrator fitted on the other folds, giving
    honest validation metrics for calibrated variants on the CAL split.

    Args:
        y_true: Binary labels.
        y_prob: Uncalibrated probabilities.
        method: Calibration method.
        n_splits: Number of folds.
        seed: Shuffle seed for the fold assignment.

    Returns:
        Out-of-fold calibrated probabilities, same shape as ``y_prob``.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    oof = np.empty_like(p)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(p, y):
        calibrator = make_calibrator(method).fit(y[train_idx], p[train_idx])
        oof[valid_idx] = calibrator.predict(p[valid_idx])
    return oof


@dataclass
class CalibratorSelection:
    """Outcome of :func:`select_calibrator`."""

    method: CalibrationMethod
    calibrator: Calibrator
    cv_brier: dict[str, float]
    oof_prob: NDArray[np.float64] = field(repr=False)


def select_calibrator(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    methods: Sequence[CalibrationMethod | str] = (
        CalibrationMethod.PLATT,
        CalibrationMethod.ISOTONIC,
    ),
    n_splits: int = 5,
    seed: int = 42,
) -> CalibratorSelection:
    """Choose the method with the lowest cross-fitted Brier score and refit it on all rows.

    Args:
        y_true: Binary labels (CAL split).
        y_prob: Uncalibrated probabilities (CAL split).
        methods: Candidate methods (ties resolved by the order given).
        n_splits: Number of cross-fitting folds.
        seed: Fold-assignment seed.

    Returns:
        The selected method, its calibrator refitted on all rows, per-method cross-fitted
        Brier scores and the winning method's out-of-fold probabilities.
    """
    y, p = check_binary_inputs(y_true, y_prob)
    cv_brier: dict[str, float] = {}
    oof_by_method: dict[CalibrationMethod, NDArray[np.float64]] = {}
    for raw_method in methods:
        method = CalibrationMethod(raw_method)
        oof = cross_fit_calibrated(y, p, method, n_splits, seed)
        oof_by_method[method] = oof
        cv_brier[method.value] = float(np.mean((oof - y) ** 2))
    best = min(oof_by_method, key=lambda m: cv_brier[m.value])
    logger.info(f"Calibrator selection (cross-fitted Brier): {cv_brier} -> {best.value}")
    return CalibratorSelection(
        method=best,
        calibrator=make_calibrator(best).fit(y, p),
        cv_brier=cv_brier,
        oof_prob=oof_by_method[best],
    )
