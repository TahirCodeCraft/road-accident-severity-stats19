"""Cleaning of the raw tables: missing-code unification, validity checks, parsing and targets."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from src.data.codebook import Codebook
from src.data.schema import (
    EVAL_ONLY_RENAMES,
    SEVERITY3_COL,
    TARGET_COL,
    columns_by_decision,
    get_spec,
    kept_columns,
    tier_b_columns,
)

logger = logging.getLogger(__name__)

VALID_SPEED_LIMITS: frozenset[int] = frozenset({20, 30, 40, 50, 60, 70})
SEVERE_CODES: frozenset[int] = frozenset({1, 2})
VALID_SEVERITY_CODES: frozenset[int] = frozenset({1, 2, 3})
DATE_FORMAT = "%d/%m/%Y"
TIME_PATTERN = r"^(\d{1,2}):(\d{2})$"
NOT_AT_JUNCTION = 0
# Columns whose -1 is structural (not at a junction) when junction_detail == 0.
STRUCTURAL_JUNCTION_COLUMNS: tuple[str, ...] = ("junction_control", "second_road_class")
STRING_MISSING_TOKENS: tuple[str, ...] = ("-1", "")


@dataclass
class ValidityReport:
    """Counts produced by the collision validity checks.

    Attributes:
        n_rows: Number of collision rows.
        n_duplicate_index: Rows whose ``collision_index`` is duplicated.
        n_invalid_date: Dates that could not be parsed.
        n_year_mismatch: Rows whose date year differs from ``collision_year``.
        n_invalid_time: Times that could not be parsed to a valid hour/minute.
        n_out_of_range_speed: Non-missing speed limits outside 20-70 mph (set to NaN).
        n_structural_fill: ``{column: rows}`` filled with "not at junction".
        n_missing_mapped: ``{column: rows}`` mapped from a missing code to NaN.
    """

    n_rows: int = 0
    n_duplicate_index: int = 0
    n_invalid_date: int = 0
    n_year_mismatch: int = 0
    n_invalid_time: int = 0
    n_out_of_range_speed: int = 0
    n_structural_fill: dict[str, int] = field(default_factory=dict)
    n_missing_mapped: dict[str, int] = field(default_factory=dict)


def fill_structural_junction(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Set structural ``-1`` to "not at junction" (0) where ``junction_detail == 0``.

    ``junction_control`` and ``second_road_class`` are not applicable away from a junction; the
    data records them as -1 (~42% of rows), which is structural, not missing.

    Args:
        frame: Raw collision frame (before missing-code mapping).

    Returns:
        Tuple of (copy of frame with fills applied, ``{column: rows filled}``).
    """
    out = frame.copy()
    not_junction = out["junction_detail"].eq(NOT_AT_JUNCTION).fillna(False)
    counts: dict[str, int] = {}
    for col in STRUCTURAL_JUNCTION_COLUMNS:
        mask = not_junction & out[col].eq(-1).fillna(False)
        out.loc[mask, col] = NOT_AT_JUNCTION
        counts[col] = int(mask.sum())
    return out, counts


def map_missing_codes(
    frame: pd.DataFrame, table: str, codebook: Codebook, columns: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace -1 and data-guide unknown codes with NA in the given columns.

    Integer columns use the codebook's per-field missing codes (always including -1); string
    columns map the tokens ``"-1"`` and ``""``. Float columns (coordinates) are left untouched
    because -1.0 can be a valid longitude.

    Args:
        frame: Frame to clean.
        table: Table the columns come from (codes are table specific).
        codebook: Parsed data guide.
        columns: Columns to process.

    Returns:
        Tuple of (cleaned copy, ``{column: rows mapped to NA}``).
    """
    out = frame.copy()
    counts: dict[str, int] = {}
    for col in columns:
        dtype = get_spec(table, col).dtype
        if dtype == "Int32":
            codes = sorted(codebook.missing_for(table, col))
            mask = out[col].isin(codes).fillna(False)
        elif dtype == "string":
            mask = out[col].isin(STRING_MISSING_TOKENS).fillna(False)
        else:
            continue
        out.loc[mask, col] = pd.NA
        counts[col] = int(mask.sum())
    return out, counts


def check_speed_limits(speed: pd.Series) -> tuple[pd.Series, int]:
    """Set non-missing speed limits outside the legal set {20..70} to NA.

    Args:
        speed: Speed-limit column (missing codes already mapped).

    Returns:
        Tuple of (cleaned series, number of out-of-range values).
    """
    invalid = speed.notna() & ~speed.isin(sorted(VALID_SPEED_LIMITS)).fillna(False)
    return speed.mask(invalid), int(invalid.sum())


def parse_dates(date: pd.Series) -> pd.Series:
    """Parse ``dd/mm/yyyy`` strings to datetimes (unparseable -> NaT)."""
    return pd.to_datetime(date, format=DATE_FORMAT, errors="coerce")


def date_parts(dates: pd.Series) -> pd.DataFrame:
    """Derive month, day-of-year and ISO week number from parsed dates.

    The ISO week is the week number of the date itself (early-January / late-December dates may
    belong to the adjacent ISO year; only the number is used).

    Args:
        dates: Datetime series.

    Returns:
        DataFrame with ``month``, ``day_of_year`` and ``iso_week`` (nullable ``Int16``).
    """
    iso = dates.dt.isocalendar()
    return pd.DataFrame(
        {
            "month": dates.dt.month.astype("Int16"),
            "day_of_year": dates.dt.day_of_year.astype("Int16"),
            "iso_week": iso["week"].astype("Int16"),
        },
        index=dates.index,
    )


def parse_hour(time: pd.Series) -> pd.Series:
    """Parse ``HH:MM`` strings to the hour (``Int16``); invalid or missing times -> NA.

    Args:
        time: String series of times.

    Returns:
        Hour of day 0-23 as nullable ``Int16``.
    """
    parts = time.astype("string").str.strip().str.extract(TIME_PATTERN)
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce")
    valid = hour.between(0, 23) & minute.between(0, 59)
    return hour.where(valid).astype("Int16")


def make_targets(severity: pd.Series) -> pd.DataFrame:
    """Build the binary target (Fatal/Serious = 1, Slight = 0) and the 3-class severity.

    Args:
        severity: Raw ``collision_severity`` codes (1 Fatal, 2 Serious, 3 Slight).

    Returns:
        DataFrame with ``target`` (int8) and ``severity3`` (int8).

    Raises:
        ValueError: If any severity is missing or outside {1, 2, 3}.
    """
    bad = severity.isna() | ~severity.isin(sorted(VALID_SEVERITY_CODES)).fillna(False)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows with invalid collision_severity")
    return pd.DataFrame(
        {
            TARGET_COL: severity.isin(sorted(SEVERE_CODES)).astype("int8"),
            SEVERITY3_COL: severity.astype("int8"),
        },
        index=severity.index,
    )


def count_duplicates(index: pd.Series) -> int:
    """Return the number of rows whose id appears more than once."""
    return int(index.duplicated(keep=False).sum())


def collision_feature_columns() -> list[str]:
    """Return kept collision columns that are carried as-is (date/time are parsed instead)."""
    return [c for c in kept_columns("collision") if get_spec("collision", c).kind != "temporal"]


def _log_report(report: ValidityReport) -> None:
    """Log a validity report."""
    logger.info(
        f"Collision validity: rows={report.n_rows:,} dup_index={report.n_duplicate_index} "
        f"invalid_date={report.n_invalid_date} year_mismatch={report.n_year_mismatch} "
        f"invalid_time={report.n_invalid_time} speed_out_of_range={report.n_out_of_range_speed}"
    )
    logger.info(f"Structural 'not at junction' fills: {report.n_structural_fill}")
    mapped = {k: v for k, v in report.n_missing_mapped.items() if v}
    logger.info(f"Collision missing codes -> NaN (rows per column): {mapped}")


def clean_collisions(raw: pd.DataFrame, codebook: Codebook) -> tuple[pd.DataFrame, ValidityReport]:
    """Clean the collision table and derive temporal fields and targets.

    Args:
        raw: Raw collision frame (all years).
        codebook: Parsed data guide.

    Returns:
        Tuple of (clean frame, validity report). Columns: ``collision_index, collision_year,
        date, hour, month, day_of_year, iso_week, target, severity3, police_attended``, then the
        kept collision columns (with ``collision_injury_based`` first).

    Raises:
        ValueError: On duplicated ``collision_index`` or unparseable dates.
    """
    report = ValidityReport(n_rows=len(raw))
    report.n_duplicate_index = count_duplicates(raw["collision_index"])
    if report.n_duplicate_index:
        raise ValueError(f"{report.n_duplicate_index} rows with duplicated collision_index")

    frame, report.n_structural_fill = fill_structural_junction(raw)
    feature_cols = collision_feature_columns()
    eval_cols = columns_by_decision("collision", "eval_only")
    frame, report.n_missing_mapped = map_missing_codes(
        frame, "collision", codebook, feature_cols + eval_cols
    )
    frame["speed_limit"], report.n_out_of_range_speed = check_speed_limits(frame["speed_limit"])

    dates = parse_dates(frame["date"])
    report.n_invalid_date = int(dates.isna().sum())
    if report.n_invalid_date:
        raise ValueError(f"{report.n_invalid_date} unparseable dates")
    report.n_year_mismatch = int((dates.dt.year != frame["collision_year"]).sum())
    hour = parse_hour(frame["time"])
    report.n_invalid_time = int(hour.isna().sum())

    first = "collision_injury_based"
    ordered = [first] + [c for c in feature_cols if c != first]
    out = pd.concat(
        [
            frame[["collision_index", "collision_year"]],
            dates.rename("date"),
            hour.rename("hour"),
            date_parts(dates),
            make_targets(frame["collision_severity"]),
            frame[eval_cols].rename(columns=EVAL_ONLY_RENAMES),
            frame[ordered],
        ],
        axis=1,
    )
    _log_report(report)
    return out, report


def clean_vehicles(raw: pd.DataFrame, codebook: Codebook) -> pd.DataFrame:
    """Clean the vehicle table: keep ids + kept + Tier-B fields, map missing codes to NA.

    Args:
        raw: Raw vehicle frame (all years).
        codebook: Parsed data guide.

    Returns:
        Vehicle-level frame with ``collision_index, collision_year, vehicle_reference`` and the
        kept and Tier-B vehicle fields.
    """
    value_cols = kept_columns("vehicle") + tier_b_columns("vehicle")
    id_cols = ["collision_index", "collision_year", "vehicle_reference"]
    frame, counts = map_missing_codes(raw[id_cols + value_cols], "vehicle", codebook, value_cols)
    logger.info(f"Vehicle missing codes -> NaN (rows per column): {counts}")
    frame = harmonise_vehicle_codes(frame)
    return frame.reset_index(drop=True)


# Codes introduced by the 2024 STATS19 specification that have no 2023 (FIT) equivalent; they are
# folded into "missing" so every year is coded the way the training year was (see report §4).
SPEC2024_TO_MISSING: dict[str, tuple[int, ...]] = {
    "journey_purpose_of_driver": (8, 9),
    "vehicle_manoeuvre": (20,),
}


def harmonise_vehicle_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """Map 2024-specification-only vehicle codes to NA to keep coding consistent across years.

    Args:
        frame: Cleaned vehicle frame.

    Returns:
        Frame with the codes in :data:`SPEC2024_TO_MISSING` replaced by NA.
    """
    frame = frame.copy()
    for col, codes in SPEC2024_TO_MISSING.items():
        if col in frame.columns:
            mask = frame[col].isin(codes)
            frame.loc[mask, col] = pd.NA
            logger.info(f"Harmonised {int(mask.sum())} rows of {col} codes {codes} -> NA")
    return frame


def clean_casualties(raw: pd.DataFrame, codebook: Codebook) -> pd.DataFrame:
    """Clean the casualty table down to the fields allowed for pedestrian features.

    Args:
        raw: Raw casualty frame (all years).
        codebook: Parsed data guide.

    Returns:
        Frame with ``collision_index, casualty_reference`` and the kept casualty fields.
    """
    value_cols = kept_columns("casualty")
    id_cols = ["collision_index", "casualty_reference"]
    frame, counts = map_missing_codes(raw[id_cols + value_cols], "casualty", codebook, value_cols)
    logger.info(f"Casualty missing codes -> NaN (rows per column): {counts}")
    return frame.reset_index(drop=True)
