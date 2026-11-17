"""Temporal split assignment (FIT / TUNE / CAL / TEST) and the guarded split loader.

FIT = 2023; TUNE = 2024 odd ISO week; CAL = 2024 even ISO week; TEST = 2025. The ISO week is the
week number of the date itself, so e.g. 30-31 Dec 2024 (ISO week 1 of 2025) fall in TUNE.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.schema import TARGET_COL
from src.paths import COLLISIONS_PATH, FROZEN_MANIFEST_PATH

logger = logging.getLogger(__name__)

SPLIT_COL = "split"
SPLITS: tuple[str, ...] = ("fit", "tune", "cal", "test")
FIT_YEAR = 2023
DEV_YEAR = 2024
TEST_YEAR = 2025


class TestSplitLockedError(PermissionError):
    """Raised when the 2025 test split is requested before the experiment is frozen."""

    __test__ = False  # not a pytest test class


def assign_split(df: pd.DataFrame) -> pd.Series:
    """Assign each collision to a split.

    Args:
        df: Frame with ``collision_year`` and ``iso_week`` columns.

    Returns:
        String series (``fit``/``tune``/``cal``/``test``) aligned with ``df``.

    Raises:
        ValueError: If any row has a year outside 2023-2025 or a missing ISO week in 2024.
    """
    year = df["collision_year"]
    week = df["iso_week"]
    dev = year.eq(DEV_YEAR)
    if (dev & week.isna()).any():
        raise ValueError("Missing iso_week for 2024 rows")
    split = pd.Series(pd.NA, index=df.index, dtype="string", name=SPLIT_COL)
    split[year.eq(FIT_YEAR)] = "fit"
    split[dev & week.mod(2).eq(1)] = "tune"
    split[dev & week.mod(2).eq(0)] = "cal"
    split[year.eq(TEST_YEAR)] = "test"
    if split.isna().any():
        raise ValueError(f"{int(split.isna().sum())} rows outside years 2023-2025")
    return split


def check_test_access(allow_test: bool, manifest_path: Path = FROZEN_MANIFEST_PATH) -> None:
    """Raise unless test access is explicitly allowed and the frozen manifest exists.

    Args:
        allow_test: Caller's explicit opt-in.
        manifest_path: Path of ``experiments/frozen_manifest.json``.

    Raises:
        TestSplitLockedError: If access is not permitted.
    """
    if not allow_test:
        raise TestSplitLockedError("Test split is locked: pass allow_test=True after freezing")
    if not manifest_path.exists():
        raise TestSplitLockedError(f"Test split is locked: {manifest_path} does not exist")


def load_split(
    name: str,
    allow_test: bool = False,
    path: Path = COLLISIONS_PATH,
    manifest_path: Path = FROZEN_MANIFEST_PATH,
) -> pd.DataFrame:
    """Load one split of the processed collision table.

    Args:
        name: One of ``fit``, ``tune``, ``cal``, ``test`` (or ``fit+tune`` style joins with
            ``+``).
        allow_test: Must be True (and the frozen manifest must exist) to load ``test``.
        path: Processed collisions parquet.
        manifest_path: Frozen manifest path checked by the test guard.

    Returns:
        Rows of the requested split(s) with a ``split`` column, index reset.

    Raises:
        ValueError: For unknown split names.
        TestSplitLockedError: When ``test`` is requested without permission.
    """
    names = name.split("+")
    unknown = sorted(set(names) - set(SPLITS))
    if unknown:
        raise ValueError(f"Unknown split(s) {unknown}; expected {SPLITS}")
    if "test" in names:
        check_test_access(allow_test, manifest_path)
    frame = pd.read_parquet(path)
    frame[SPLIT_COL] = assign_split(frame)
    out = frame.loc[frame[SPLIT_COL].isin(names)].reset_index(drop=True)
    logger.info(f"Loaded split {name}: {len(out):,} rows")
    return out


def split_summary(df: pd.DataFrame, include_test_labels: bool = False) -> pd.DataFrame:
    """Summarise rows and severe rate per split.

    Args:
        df: Collision frame with ``collision_year``, ``iso_week`` and ``target``.
        include_test_labels: Report TEST label statistics (only after the freeze).

    Returns:
        DataFrame with ``split, rows, n_severe, severe_rate`` in split order (TEST label
        statistics are NaN unless ``include_test_labels``).
    """
    split = assign_split(df)
    grouped = df[TARGET_COL].groupby(split)
    summary = pd.DataFrame(
        {"rows": grouped.size(), "n_severe": grouped.sum(), "severe_rate": grouped.mean()}
    ).reindex(list(SPLITS))
    summary["severe_rate"] = summary["severe_rate"].round(4)
    if not include_test_labels:
        summary.loc["test", ["n_severe", "severe_rate"]] = float("nan")
    summary.index.name = SPLIT_COL
    return summary.reset_index()
