"""Project directory layout, derived from the location of this file (no hardcoded paths)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
TABLES_DIR: Path = REPORTS_DIR / "tables"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

DATA_GUIDE_PATH: Path = RAW_DIR / "data-guide-2025.xlsx"
COLLISIONS_PATH: Path = PROCESSED_DIR / "collisions.parquet"
VEHICLES_CLEAN_PATH: Path = PROCESSED_DIR / "vehicles_clean.parquet"
FROZEN_MANIFEST_PATH: Path = EXPERIMENTS_DIR / "frozen_manifest.json"

RAW_FILE_TEMPLATE = "dft-road-casualty-statistics-{table}-{year}.csv"


def raw_csv_path(table: str, year: int, raw_dir: Path = RAW_DIR) -> Path:
    """Return the path of one raw STATS19 CSV.

    Args:
        table: One of ``"collision"``, ``"vehicle"``, ``"casualty"``.
        year: Data year (e.g. 2023).
        raw_dir: Directory holding the raw CSVs.

    Returns:
        Path to the CSV file.
    """
    return raw_dir / RAW_FILE_TEMPLATE.format(table=table, year=year)
