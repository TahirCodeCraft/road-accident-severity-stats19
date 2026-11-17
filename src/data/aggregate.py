"""Collision-level, order-free aggregation of the vehicle and casualty tables.

Vehicle rows are summarised with permutation-invariant statistics only (counts per code, min/mean/
max), so the arbitrary ``vehicle_reference`` order never leaks into features. Casualty rows exist
only for injured people, so the only allowed signal is the pedestrian flag plus the age, location
and movement of the first pedestrian (lowest ``casualty_reference``); no casualty counts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.data.schema import (
    N_VEHICLE_ROWS_COL,
    PED_FEATURES,
    PED_SOURCE_COLUMNS,
    TIERB_PREFIX,
    VEHICLE_NUMERIC_STATS,
    VEHICLE_PREFIX,
    get_spec,
    kept_columns,
    tier_b_columns,
)

logger = logging.getLogger(__name__)

KEY = "collision_index"
OTHER_LEVEL = "other"
MISSING_LEVEL = "missing"
PEDESTRIAN_CLASS = 3
COUNT_DTYPE = "int16"


@dataclass(frozen=True)
class VehicleAggConfig:
    """Configuration of the vehicle aggregation.

    Attributes:
        fit_year: Year whose vehicle rows define the code levels (others -> ``__other``).
        count_prefix: Prefix of default vehicle count columns.
        tierb_prefix: Prefix of Tier-B count columns (deny-listed from default features).
    """

    fit_year: int = 2023
    count_prefix: str = VEHICLE_PREFIX
    tierb_prefix: str = TIERB_PREFIX


def vehicle_count_fields() -> list[str]:
    """Return kept low-cardinality (categorical) vehicle fields summarised by code counts."""
    return [c for c in kept_columns("vehicle") if get_spec("vehicle", c).kind == "categorical"]


def vehicle_numeric_fields() -> list[str]:
    """Return kept numeric vehicle fields summarised by min/mean/max."""
    return [c for c in kept_columns("vehicle") if get_spec("vehicle", c).kind == "numeric"]


def fit_code_levels(
    vehicles: pd.DataFrame, fields: Sequence[str], fit_year: int
) -> dict[str, list[int]]:
    """Return the sorted non-missing codes of each field observed in the FIT year.

    Args:
        vehicles: Clean vehicle-level frame (must contain ``collision_year``).
        fields: Fields to compute levels for.
        fit_year: The FIT year (2023).

    Returns:
        ``{field: sorted codes}``.

    Raises:
        ValueError: If no rows belong to ``fit_year``.
    """
    fit_rows = vehicles.loc[vehicles["collision_year"].eq(fit_year).fillna(False)]
    if fit_rows.empty:
        raise ValueError(f"No vehicle rows for fit year {fit_year}")
    return {f: sorted(int(v) for v in fit_rows[f].dropna().unique()) for f in fields}


def code_labels(codes: pd.Series, levels: Sequence[int]) -> pd.Series:
    """Map codes to string level labels: the code itself, ``other`` or ``missing``.

    Args:
        codes: Integer code series (NA = missing).
        levels: Codes seen in the FIT year.

    Returns:
        String series of level labels.
    """
    labels = codes.astype("string")
    unseen = codes.notna() & ~codes.isin(list(levels)).fillna(False)
    labels = labels.mask(unseen, OTHER_LEVEL)
    return labels.fillna(MISSING_LEVEL)


def count_codes(
    vehicles: pd.DataFrame, field: str, levels: Sequence[int], prefix: str
) -> pd.DataFrame:
    """Count vehicle rows per code of one field for each collision.

    Args:
        vehicles: Clean vehicle-level frame.
        field: Field to summarise.
        levels: FIT-year codes; one column per level plus ``__other`` and ``__missing``.
        prefix: Column prefix (``veh_`` or ``tierb_``).

    Returns:
        Frame indexed by ``collision_index`` with columns ``<prefix><field>__<level>``.
    """
    labels = code_labels(vehicles[field], levels).rename("level")
    counts = vehicles[[KEY]].assign(level=labels).groupby([KEY, "level"]).size()
    wide = counts.unstack("level", fill_value=0)
    columns = [str(level) for level in levels] + [OTHER_LEVEL, MISSING_LEVEL]
    wide = wide.reindex(columns=columns, fill_value=0).astype(COUNT_DTYPE)
    wide.columns = [f"{prefix}{field}__{level}" for level in columns]
    wide.columns.name = None
    return wide


def numeric_summaries(vehicles: pd.DataFrame, fields: Sequence[str], prefix: str) -> pd.DataFrame:
    """Compute min/mean/max of numeric vehicle fields per collision (NaN if all missing).

    Args:
        vehicles: Clean vehicle-level frame.
        fields: Numeric fields.
        prefix: Column prefix.

    Returns:
        Frame indexed by ``collision_index`` with ``<prefix><field>_<stat>`` float columns.
    """
    values = vehicles[[KEY, *fields]].copy()
    values[list(fields)] = values[list(fields)].astype("float64")
    stats = values.groupby(KEY)[list(fields)].agg(list(VEHICLE_NUMERIC_STATS))
    stats.columns = [f"{prefix}{field}_{stat}" for field, stat in stats.columns]
    return stats.astype("float64")


def aggregate_vehicles(
    vehicles: pd.DataFrame, config: VehicleAggConfig | None = None
) -> pd.DataFrame:
    """Aggregate clean vehicle rows to one row per collision (order-free summaries only).

    Args:
        vehicles: Clean vehicle-level frame from :func:`src.data.clean.clean_vehicles`.
        config: Aggregation configuration (defaults to :class:`VehicleAggConfig`).

    Returns:
        Frame indexed by ``collision_index`` with ``n_vehicle_rows``, ``veh_*`` counts and
        numeric summaries, and ``tierb_*`` counts.

    Raises:
        ValueError: If a ``(collision_index, vehicle_reference)`` key is duplicated.
    """
    if "vehicle_reference" in vehicles.columns:
        dup = int(vehicles.duplicated(["collision_index", "vehicle_reference"]).sum())
        if dup:
            raise ValueError(f"{dup} duplicated (collision_index, vehicle_reference) keys")
    config = config or VehicleAggConfig()
    count_fields = vehicle_count_fields()
    tierb_fields = tier_b_columns("vehicle")
    levels = fit_code_levels(vehicles, count_fields + tierb_fields, config.fit_year)
    for field, codes in levels.items():
        logger.debug(f"FIT levels {field}: {codes}")
    n_rows = vehicles.groupby(KEY).size().rename(N_VEHICLE_ROWS_COL).astype(COUNT_DTYPE)
    parts = [n_rows.to_frame()]
    parts += [count_codes(vehicles, f, levels[f], config.count_prefix) for f in count_fields]
    parts.append(numeric_summaries(vehicles, vehicle_numeric_fields(), config.count_prefix))
    parts += [count_codes(vehicles, f, levels[f], config.tierb_prefix) for f in tierb_fields]
    result = pd.concat(parts, axis=1).sort_index()
    logger.info(
        f"Vehicle aggregates: {len(result):,} collisions x {result.shape[1]} columns "
        f"({len(count_fields)} count fields, {len(tierb_fields)} Tier-B fields)"
    )
    return result


def pedestrian_features(casualties: pd.DataFrame) -> pd.DataFrame:
    """Derive the pedestrian flag and first-pedestrian attributes per collision.

    Args:
        casualties: Clean casualty frame (``collision_index``, ``casualty_reference``,
            ``casualty_class``, ``age_of_casualty``, ``pedestrian_location``,
            ``pedestrian_movement``).

    Returns:
        Frame indexed by ``collision_index`` (collisions with any casualty row) with
        ``ped_involved`` (int8) and ``ped_age``, ``ped_location``, ``ped_movement`` of the
        pedestrian with the lowest ``casualty_reference`` (NA when no pedestrian).
    """
    is_ped = casualties["casualty_class"].eq(PEDESTRIAN_CLASS).fillna(False)
    flag = is_ped.groupby(casualties[KEY]).any().astype("int8").rename("ped_involved")
    source_cols = list(PED_SOURCE_COLUMNS)
    peds = casualties.loc[is_ped, [KEY, "casualty_reference", *source_cols]]
    peds = peds.sort_values([KEY, "casualty_reference", *source_cols], kind="mergesort")
    first = peds.drop_duplicates(KEY, keep="first").set_index(KEY)[source_cols]
    first = first.rename(columns=PED_SOURCE_COLUMNS)
    return flag.to_frame().join(first, how="left")[list(PED_FEATURES)].sort_index()


def attach_aggregates(
    collisions: pd.DataFrame, vehicle_agg: pd.DataFrame, ped: pd.DataFrame
) -> pd.DataFrame:
    """Left-join vehicle aggregates and pedestrian fields onto the collision table.

    Count columns of collisions without vehicle rows are filled with 0, ``ped_involved`` with 0.

    Args:
        collisions: Clean collision frame (one row per ``collision_index``).
        vehicle_agg: Output of :func:`aggregate_vehicles`.
        ped: Output of :func:`pedestrian_features`.

    Returns:
        Collision frame with the extra columns; row count unchanged.

    Raises:
        AssertionError: If the join changes the number of rows.
    """
    n_before = len(collisions)
    out = collisions.merge(
        vehicle_agg, how="left", left_on=KEY, right_index=True, validate="one_to_one"
    )
    assert len(out) == n_before, "vehicle join changed row count"
    out = out.merge(ped, how="left", left_on=KEY, right_index=True, validate="one_to_one")
    assert len(out) == n_before, "casualty join changed row count"
    count_cols = [c for c in vehicle_agg.columns if "__" in c or c == N_VEHICLE_ROWS_COL]
    out[count_cols] = out[count_cols].fillna(0).astype(COUNT_DTYPE)
    out["ped_involved"] = out["ped_involved"].fillna(0).astype("int8")
    return out
