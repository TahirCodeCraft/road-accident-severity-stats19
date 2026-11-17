"""Read the raw STATS19 CSVs with explicit dtypes taken from the column registry."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.data.schema import TABLES, raw_columns, read_dtypes
from src.paths import RAW_DIR, raw_csv_path

logger = logging.getLogger(__name__)

DEFAULT_YEARS: tuple[int, ...] = (2023, 2024, 2025)


def check_columns(frame: pd.DataFrame, table: str, source: Path) -> None:
    """Raise if the CSV columns differ from the registered schema.

    Args:
        frame: Loaded frame.
        table: Table name.
        source: File path (for the error message).

    Raises:
        ValueError: On unregistered or missing columns.
    """
    expected = raw_columns(table)
    extra = sorted(set(frame.columns) - set(expected))
    missing = sorted(set(expected) - set(frame.columns))
    if extra or missing:
        raise ValueError(f"{source.name}: unregistered columns {extra}, missing columns {missing}")


def read_table_year(table: str, year: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read one raw CSV (one table, one year) with explicit dtypes.

    Integer codes are read as nullable ``Int32``; ``collision_index`` and other identifiers or
    string categories as ``string``.

    Args:
        table: One of :data:`src.data.schema.TABLES`.
        year: Data year.
        raw_dir: Directory holding the raw CSVs.

    Returns:
        The loaded frame.
    """
    path = raw_csv_path(table, year, raw_dir)
    frame = pd.read_csv(path, dtype=read_dtypes(table))
    check_columns(frame, table, path)
    logger.info(f"Loaded {path.name}: {len(frame):,} rows x {frame.shape[1]} cols")
    return frame


def read_table(
    table: str, years: Sequence[int] = DEFAULT_YEARS, raw_dir: Path = RAW_DIR
) -> pd.DataFrame:
    """Read and concatenate one table across years.

    Args:
        table: Table name.
        years: Years to read.
        raw_dir: Directory holding the raw CSVs.

    Returns:
        Concatenated frame with a fresh RangeIndex.
    """
    frames = [read_table_year(table, year, raw_dir) for year in years]
    return pd.concat(frames, ignore_index=True)


def read_all(
    years: Sequence[int] = DEFAULT_YEARS, raw_dir: Path = RAW_DIR
) -> dict[str, pd.DataFrame]:
    """Read all three tables (9 CSVs for 2023-2025).

    Args:
        years: Years to read.
        raw_dir: Directory holding the raw CSVs.

    Returns:
        ``{table: frame}`` for collision, vehicle and casualty.
    """
    return {table: read_table(table, years, raw_dir) for table in TABLES}
