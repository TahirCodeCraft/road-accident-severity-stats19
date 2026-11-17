"""Shared input validation for binary-classification evaluation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_binary_inputs(
    y_true: ArrayLike, y_prob: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Validate and convert binary labels and positive-class probabilities.

    Args:
        y_true: Binary labels in {0, 1}, shape (n,).
        y_prob: Predicted probabilities of the positive class in [0, 1], shape (n,).

    Returns:
        Tuple ``(y, p)`` as contiguous int64 and float64 1-D arrays.

    Raises:
        ValueError: On shape mismatch, empty input, non-binary labels, or invalid probabilities.
    """
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_prob lengths differ: {y.shape[0]} vs {p.shape[0]}")
    if y.size == 0:
        raise ValueError("Empty input arrays")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y_true must contain only 0/1 labels")
    if not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("y_prob must be finite and lie in [0, 1]")
    return y.astype(np.int64), p


def check_binary_predictions(y_pred: ArrayLike, n: int) -> NDArray[np.int64]:
    """Validate hard 0/1 predictions of a given length.

    Args:
        y_pred: Binary predictions, shape (n,).
        n: Expected length.

    Returns:
        Predictions as an int64 1-D array.

    Raises:
        ValueError: On length mismatch or non-binary values.
    """
    pred = np.asarray(y_pred).ravel()
    if pred.shape[0] != n:
        raise ValueError(f"Prediction length {pred.shape[0]} does not match labels length {n}")
    if not np.all((pred == 0) | (pred == 1)):
        raise ValueError("Predictions must contain only 0/1 values")
    return pred.astype(np.int64)
