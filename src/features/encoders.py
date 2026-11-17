"""Fitted categorical encoders: frequency encoding and out-of-fold m-estimate target encoding.

Both encoders follow a ``fit`` / ``transform`` API and must be fitted on training rows only
(FIT in the ``tune`` phase, FIT+TUNE in the ``refit`` phase). Missing keys and keys unseen at fit
time map to ``0`` (frequency) or the training prior (target encoding).

Target encoding uses m-estimate smoothing::

    te(k) = (sum_y(k) + m * prior) / (n(k) + m)

For the training rows themselves, :meth:`OOFTargetEncoder.fit_transform` returns out-of-fold
values (shuffled K folds, seeded, label-independent) so a row's own label never
contributes to its encoding;
every other row is encoded with the map fitted on all training rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


def _as_key(values: pd.Series) -> pd.Series:
    """Normalise a key series to ``object`` with ``None`` for missing (hashable, NA-safe)."""
    keys = values.astype(object)
    return keys.where(values.notna(), None)


@dataclass
class FrequencyEncoder:
    """Encode a categorical key by its relative frequency among the training rows.

    Attributes:
        name: Column name (for logging).
        frequencies: Learned ``{key: share of training rows}``.
        n_train: Number of training rows seen at fit time.
    """

    name: str = ""
    frequencies: dict[object, float] = field(default_factory=dict)
    n_train: int = 0

    def fit(self, keys: pd.Series) -> FrequencyEncoder:
        """Learn key frequencies from training rows.

        Args:
            keys: Key series of the training rows (NA keys are ignored).

        Returns:
            The fitted encoder.
        """
        clean = _as_key(keys).dropna()
        self.n_train = len(keys)
        counts = clean.value_counts()
        self.frequencies = (counts / max(self.n_train, 1)).to_dict()
        logger.debug(f"FrequencyEncoder[{self.name}] fitted on {self.n_train:,} rows, "
                     f"{len(self.frequencies)} keys")
        return self

    def transform(self, keys: pd.Series) -> pd.Series:
        """Map keys to training frequencies (missing / unseen -> 0).

        Args:
            keys: Key series of any rows.

        Returns:
            float32 series aligned with ``keys``.
        """
        mapped = _as_key(keys).map(self.frequencies)
        return mapped.astype("float64").fillna(0.0).astype(np.float32)


@dataclass
class OOFTargetEncoder:
    """Out-of-fold m-estimate target encoder for one categorical key.

    Attributes:
        name: Column name (for logging).
        m: Smoothing strength (pseudo-count pulled towards the prior).
        n_splits: Number of out-of-fold splits for training rows.
        seed: Seed of the fold assignment.
        prior: Training-row positive rate (set by :meth:`fit`).
        mapping: Full-training-rows ``{key: encoded value}`` (set by :meth:`fit`).
    """

    name: str = ""
    m: float = 50.0
    n_splits: int = 5
    seed: int = 42
    prior: float = float("nan")
    mapping: dict[object, float] = field(default_factory=dict)

    @staticmethod
    def _smoothed_map(keys: pd.Series, y: np.ndarray, prior: float, m: float
                      ) -> dict[object, float]:
        """Compute the m-estimate map ``{key: (sum + m*prior) / (n + m)}`` (NA keys ignored)."""
        frame = pd.DataFrame({"k": _as_key(keys).to_numpy(), "y": y}).dropna(subset=["k"])
        stats = frame.groupby("k", sort=False)["y"].agg(["sum", "count"])
        values = (stats["sum"] + m * prior) / (stats["count"] + m)
        return values.to_dict()

    def _apply(self, keys: pd.Series, mapping: dict[object, float], prior: float) -> np.ndarray:
        """Map keys with ``mapping``; missing / unseen keys get ``prior``."""
        mapped = _as_key(keys).map(mapping).astype("float64").fillna(prior)
        return mapped.to_numpy()

    def fit(self, keys: pd.Series, y: np.ndarray) -> OOFTargetEncoder:
        """Fit the full-training-rows map.

        Args:
            keys: Key series of the training rows.
            y: Binary labels of the training rows.

        Returns:
            The fitted encoder.
        """
        y = np.asarray(y, dtype=np.float64)
        self.prior = float(y.mean())
        self.mapping = self._smoothed_map(keys.reset_index(drop=True), y, self.prior, self.m)
        logger.debug(f"OOFTargetEncoder[{self.name}] prior={self.prior:.4f} "
                     f"keys={len(self.mapping)}")
        return self

    def transform(self, keys: pd.Series) -> pd.Series:
        """Encode non-training rows with the full-training-rows map.

        Args:
            keys: Key series of any rows.

        Returns:
            float32 series aligned with ``keys``.

        Raises:
            RuntimeError: If the encoder is not fitted.
        """
        if np.isnan(self.prior):
            raise RuntimeError(f"OOFTargetEncoder[{self.name}] is not fitted")
        values = self._apply(keys, self.mapping, self.prior)
        return pd.Series(values, index=keys.index, dtype=np.float32)

    def fit_transform(self, keys: pd.Series, y: np.ndarray) -> pd.Series:
        """Fit on training rows and return their out-of-fold encodings.

        Each training row is encoded with a map fitted on the other ``n_splits - 1`` folds (with
        that fold's own prior), so its own label never contributes to its value.

        Args:
            keys: Key series of the training rows.
            y: Binary labels of the training rows.

        Returns:
            float32 series of out-of-fold encodings aligned with ``keys``.
        """
        y = np.asarray(y, dtype=np.float64)
        self.fit(keys, y)
        local = keys.reset_index(drop=True)
        oof = np.empty(len(local), dtype=np.float64)
        folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for train_idx, hold_idx in folds.split(np.zeros(len(y))):
            prior = float(y[train_idx].mean())
            fold_map = self._smoothed_map(local.iloc[train_idx], y[train_idx], prior, self.m)
            oof[hold_idx] = self._apply(local.iloc[hold_idx], fold_map, prior)
        return pd.Series(oof, index=keys.index, dtype=np.float32)
