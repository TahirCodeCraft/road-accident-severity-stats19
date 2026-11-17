"""Cost-sensitive learning and resampling strategies (applied to training rows only).

Strategies (the ``CS`` factor of the ablation):

* ``none``   — natural class distribution, unit weights.
* ``weight`` — class weighting: positive weight = n_neg / n_pos (fixed, not tuned).
* ``rus``    — random undersampling of the majority class to 1:1.
* ``smotenc`` — SMOTE-NC oversampling of the minority class to 1:1 (mixed categorical/numeric).

Resampling returns a *new* training frame; TUNE / CAL / TEST rows are never touched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from src.features.adapters import split_columns

logger = logging.getLogger(__name__)

STRATEGIES = ("none", "weight", "rus", "smotenc")


@dataclass
class ImbalanceResult:
    """Training data after applying an imbalance strategy.

    Attributes:
        X: Training features (possibly resampled).
        y: Training labels (possibly resampled).
        sample_weight: Per-row weights (``None`` = unit weights).
        pos_weight: Positive-class weight implied by the strategy (1.0 if none).
        num_fill: Training medians used to impute numerics before resampling (SMOTE-NC only);
            the same values must be applied to every split at prediction time.
    """

    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray | None
    pos_weight: float
    num_fill: dict[str, float] | None = None


def positive_weight(y: np.ndarray) -> float:
    """Compute the balancing weight for the positive class.

    Args:
        y: Binary labels.

    Returns:
        ``n_negative / n_positive``.
    """
    n_pos = int(np.sum(y == 1))
    return float((len(y) - n_pos) / max(n_pos, 1))


def _undersample(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomly undersample the majority class to a 1:1 ratio."""
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=seed)
    rus.fit_resample(np.zeros((len(y), 1)), y)
    idx = np.sort(rus.sample_indices_)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def _smotenc(
    X: pd.DataFrame, y: np.ndarray, seed: int
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    """Oversample the minority class to 1:1 with SMOTE-NC.

    SMOTE-NC cannot handle NaN, so numerics are median-imputed (medians are returned so the same
    imputation is applied to every split) and categorical missingness is an explicit level.
    Numerics are standardised for the neighbour search and transformed back afterwards; synthetic
    values of integer-valued columns (counts, flags) are rounded so they stay on their support.

    Returns:
        ``(X_resampled, y_resampled, numeric_fill_values)``.
    """
    cat_cols, num_cols = split_columns(X)
    work = X.copy()
    vocab: dict[str, list] = {}
    for c in cat_cols:
        codes = work[c].cat.codes.astype(np.int32)  # -1 = missing (kept as its own level)
        vocab[c] = list(work[c].cat.categories)
        work[c] = codes
    medians = work[num_cols].median().fillna(0.0)
    work[num_cols] = work[num_cols].fillna(medians).astype(np.float64)
    integer_cols = [c for c in num_cols if np.allclose(work[c], np.round(work[c]))]
    mean, std = work[num_cols].mean(), work[num_cols].std().replace(0.0, 1.0).fillna(1.0)
    work[num_cols] = (work[num_cols] - mean) / std
    cat_idx = [work.columns.get_loc(c) for c in cat_cols]
    sampler = SMOTENC(categorical_features=cat_idx, sampling_strategy=1.0, random_state=seed,
                      k_neighbors=5)
    X_res, y_res = sampler.fit_resample(work, y)
    X_res = pd.DataFrame(X_res, columns=work.columns)
    X_res[num_cols] = X_res[num_cols].astype(np.float64) * std + mean
    for c in integer_cols:
        X_res[c] = np.round(X_res[c])
    for c in cat_cols:
        codes = X_res[c].astype(np.int32).to_numpy()
        X_res[c] = pd.Categorical.from_codes(codes, categories=vocab[c])
    for c in num_cols:
        X_res[c] = X_res[c].astype(np.float32)
    fill = {c: float(v) for c, v in medians.items()}
    return X_res[X.columns], np.asarray(y_res), fill


def apply_imbalance(X: pd.DataFrame, y: np.ndarray, strategy: str, seed: int) -> ImbalanceResult:
    """Apply a cost-sensitive / resampling strategy to the training data.

    Args:
        X: Training features.
        y: Training labels (1 = severe).
        strategy: One of :data:`STRATEGIES`.
        seed: Random seed for resampling.

    Returns:
        :class:`ImbalanceResult` with the data to fit on.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy: {strategy}")
    y = np.asarray(y).astype(int)
    if strategy == "none":
        return ImbalanceResult(X, y, None, 1.0)
    if strategy == "weight":
        w_pos = positive_weight(y)
        weights = np.where(y == 1, w_pos, 1.0).astype(np.float32)
        return ImbalanceResult(X, y, weights, w_pos)
    fill = None
    if strategy == "rus":
        X_r, y_r = _undersample(X, y, seed)
    else:
        X_r, y_r, fill = _smotenc(X, y, seed)
    logger.info("%s: %d -> %d training rows (pos rate %.3f)", strategy, len(y), len(y_r),
                float(np.mean(y_r)))
    return ImbalanceResult(X_r, y_r, None, 1.0, fill)
