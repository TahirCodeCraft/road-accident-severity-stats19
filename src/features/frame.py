"""Shared helpers for feature frames: phases, dtype coercion, assembly, validation and IO.

Feature-frame contract (consumed by ``src/training/runner.py``)::

    data/processed/features/{feature_set}_{phase}.parquet
        phase = "tune":  encoders fitted on FIT rows;       rows of splits fit / tune / cal
        phase = "refit": encoders fitted on FIT+TUNE rows;  rows of splits fit / tune / cal / test
        columns: collision_index, split, target, <features...>
        categorical features: pandas ``category``; numeric features: float32 with NaN = missing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import TARGET_COL, TIERB_PREFIX, processed_deny_list
from src.data.split import SPLIT_COL, assign_split
from src.paths import COLLISIONS_PATH, PROCESSED_DIR

logger = logging.getLogger(__name__)

KEY_COL = "collision_index"
ID_COLS: tuple[str, ...] = (KEY_COL, SPLIT_COL, TARGET_COL)
PHASES: tuple[str, ...] = ("tune", "refit")
PHASE_TRAIN_SPLITS: dict[str, tuple[str, ...]] = {"tune": ("fit",), "refit": ("fit", "tune")}
PHASE_ROW_SPLITS: dict[str, tuple[str, ...]] = {
    "tune": ("fit", "tune", "cal"),
    "refit": ("fit", "tune", "cal", "test"),
}
FEATURES_DIR: Path = PROCESSED_DIR / "features"
# Processed columns that must never appear in a feature parquet (beyond the schema deny list).
EXTRA_FORBIDDEN: frozenset[str] = frozenset({"local_authority_ons_district"})


def feature_path(feature_set: str, phase: str, features_dir: Path = FEATURES_DIR) -> Path:
    """Return the parquet path of one feature frame.

    Args:
        feature_set: ``F0``, ``F1full`` or ``F1``.
        phase: ``tune`` or ``refit``.
        features_dir: Output directory.

    Returns:
        Parquet path.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}; expected {PHASES}")
    return features_dir / f"{feature_set}_{phase}.parquet"


def load_collisions(path: Path = COLLISIONS_PATH) -> pd.DataFrame:
    """Load the processed collision table and attach the ``split`` column.

    The feature build legitimately needs TEST rows (refit-phase frames carry them, unlabelled
    use only), so this reads the parquet directly instead of the guarded ``load_split``.

    Args:
        path: Processed collisions parquet.

    Returns:
        Collision frame with a string ``split`` column.
    """
    frame = pd.read_parquet(path)
    frame[SPLIT_COL] = assign_split(frame).astype(str)
    logger.info(f"Loaded {len(frame):,} collisions from {path.name}")
    return frame


def phase_rows(collisions: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Return the rows of a phase (no TEST rows in ``tune``), index reset.

    Args:
        collisions: Collision frame with ``split``.
        phase: ``tune`` or ``refit``.

    Returns:
        Subset of rows.
    """
    keep = collisions[SPLIT_COL].isin(PHASE_ROW_SPLITS[phase])
    return collisions.loc[keep].reset_index(drop=True)


def train_mask(rows: pd.DataFrame, phase: str) -> pd.Series:
    """Return the boolean mask of the rows that encoders may be fitted on in ``phase``.

    Args:
        rows: Frame with ``split``.
        phase: ``tune`` (FIT) or ``refit`` (FIT+TUNE).

    Returns:
        Boolean series aligned with ``rows``.
    """
    return rows[SPLIT_COL].isin(PHASE_TRAIN_SPLITS[phase])


def to_float32(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert numeric / nullable / boolean values to float32 with NaN for missing.

    Args:
        values: Series or array (nullable dtypes allowed).

    Returns:
        float32 numpy array.
    """
    series = pd.Series(values)
    if isinstance(series.dtype, pd.CategoricalDtype):
        raise TypeError("to_float32 expects numeric values, got categorical")
    return series.astype("Float64").to_numpy(dtype=np.float32, na_value=np.nan)


def code_label(value: object) -> str:
    """Return the string label of a code (integral numbers without a decimal point).

    Integer-valued categoricals are stored by pyarrow as plain integers, so every categorical
    vocabulary uses string labels to survive the parquet round-trip.

    Args:
        value: Non-missing code or label.

    Returns:
        String label (``3`` -> ``"3"``, ``3.0`` -> ``"3"``, ``"A-U"`` -> ``"A-U"``).
    """
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def to_category(values: pd.Series, categories: Sequence[object]) -> pd.Categorical:
    """Convert values to a string categorical with a fixed vocabulary (other values -> NaN).

    Args:
        values: Raw values (nullable ints or strings).
        categories: Fixed, ordered vocabulary (converted with :func:`code_label`).

    Returns:
        Categorical array with string categories.
    """
    series = pd.Series(values).astype(object)
    present = series.notna()
    lookup = {v: code_label(v) for v in series[present].unique()}
    labels = series.where(~present, series[present].map(lookup)).where(present, None)
    return pd.Categorical(labels.to_numpy(), categories=[code_label(c) for c in categories])


def category_vocab(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, list[str]]:
    """Return the sorted set of non-missing values of each column (label-free vocabulary).

    Args:
        frame: Source frame (all rows; the vocabulary uses no labels).
        columns: Columns to scan.

    Returns:
        ``{column: sorted string labels}`` (integer codes sorted numerically).
    """
    vocab: dict[str, list[str]] = {}
    for col in columns:
        values = frame[col].dropna().unique()
        if pd.api.types.is_integer_dtype(frame[col].dtype):
            vocab[col] = [code_label(v) for v in sorted(int(v) for v in values)]
        else:
            vocab[col] = sorted(str(v) for v in values)
    return vocab


def forbidden_columns(columns: Iterable[str]) -> set[str]:
    """Return feature columns that are deny-listed (metadata, targets, eval-only, Tier B, raw IDs).

    Args:
        columns: Candidate feature column names (ID columns excluded).

    Returns:
        Set of offending names.
    """
    cols = [c for c in columns if c not in ID_COLS]
    bad = processed_deny_list(cols) | {c for c in cols if c in EXTRA_FORBIDDEN}
    return bad | {c for c in cols if c.startswith(TIERB_PREFIX)}


def assemble(rows: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Prepend the ID columns to a feature block and validate the contract.

    Args:
        rows: Phase rows with ``collision_index``, ``split``, ``target``.
        features: Feature block aligned with ``rows`` (same index).

    Returns:
        Frame ``collision_index, split, target, <features>``.

    Raises:
        ValueError: On deny-listed columns, misalignment or invalid dtypes.
    """
    if len(rows) != len(features) or not rows.index.equals(features.index):
        raise ValueError("Feature block is not aligned with the phase rows")
    bad = forbidden_columns(features.columns)
    if bad:
        raise ValueError(f"Deny-listed columns in features: {sorted(bad)}")
    ids = pd.DataFrame(
        {
            KEY_COL: rows[KEY_COL].astype(str).to_numpy(),
            SPLIT_COL: rows[SPLIT_COL].astype(str).to_numpy(),
            TARGET_COL: rows[TARGET_COL].astype(np.int8).to_numpy(),
        },
        index=rows.index,
    )
    out = pd.concat([ids, features], axis=1)
    validate_dtypes(out)
    return out


def validate_dtypes(frame: pd.DataFrame) -> None:
    """Assert every feature is ``category`` or float32.

    Args:
        frame: Assembled feature frame.

    Raises:
        TypeError: If a feature has another dtype.
    """
    for col in frame.columns:
        if col in ID_COLS:
            continue
        dtype = frame[col].dtype
        if not (isinstance(dtype, pd.CategoricalDtype) or dtype == np.float32):
            raise TypeError(f"Feature {col!r} has dtype {dtype}; expected category or float32")


def write_frame(frame: pd.DataFrame, feature_set: str, phase: str,
                features_dir: Path = FEATURES_DIR) -> Path:
    """Write a feature frame to parquet.

    Args:
        frame: Assembled frame.
        feature_set: Feature-set name.
        phase: Phase name.
        features_dir: Output directory.

    Returns:
        Written path.
    """
    if phase == "tune" and frame[SPLIT_COL].eq("test").any():
        raise ValueError("Tune-phase frame must not contain TEST rows")
    path = feature_path(feature_set, phase, features_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    n_feat = frame.shape[1] - len(ID_COLS)
    n_cat = sum(isinstance(frame[c].dtype, pd.CategoricalDtype) for c in frame.columns)
    logger.info(f"Wrote {path.name}: {len(frame):,} rows x {n_feat} features "
                f"({n_cat} categorical)")
    return path


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the feature columns of an assembled frame (everything but the ID columns)."""
    return [c for c in frame.columns if c not in ID_COLS]
