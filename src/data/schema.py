"""Column registry and leakage audit: the single source of truth for column decisions.

Every column of the three raw STATS19 tables (collision, vehicle, casualty) is classified with one
decision and a one-line reason, following ``docs/IMPLEMENTATION_PLAN.md`` section 1.

Decisions:
    keep: allowed predictor (or the raw source of an allowed derived predictor).
    drop_leakage: outcome-derived or outcome-conditional information; never a feature.
    drop_id: identifier / join or split key; never a feature.
    drop_other: redundant, phased out, absent in some years or too high-cardinality.
    target: the prediction target.
    tier_b: borderline crash-mechanics fields; kept in the processed table under the ``tierb_``
        prefix but deny-listed from default features (added back only in sensitivity S1).
    eval_only: kept outside X as an evaluation stratifier (police attendance).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import pandas as pd

TABLES: tuple[str, ...] = ("collision", "vehicle", "casualty")
DECISIONS: tuple[str, ...] = (
    "keep",
    "drop_leakage",
    "drop_id",
    "drop_other",
    "target",
    "tier_b",
    "eval_only",
)
NON_FEATURE_DECISIONS: frozenset[str] = frozenset(
    {"drop_leakage", "drop_id", "drop_other", "target", "tier_b", "eval_only"}
)

# Processed-table column names (collision level).
TARGET_COL = "target"
SEVERITY3_COL = "severity3"
POLICE_ATTENDED_COL = "police_attended"
EVAL_ONLY_RENAMES: dict[str, str] = {
    "did_police_officer_attend_scene_of_accident": POLICE_ATTENDED_COL,
}
META_COLUMNS: tuple[str, ...] = ("collision_index", "collision_year", "date", "iso_week", "split")
DERIVED_TEMPORAL_FEATURES: tuple[str, ...] = ("hour", "month", "day_of_year")
N_VEHICLE_ROWS_COL = "n_vehicle_rows"
VEHICLE_PREFIX = "veh_"
TIERB_PREFIX = "tierb_"
PED_FEATURES: tuple[str, ...] = ("ped_involved", "ped_age", "ped_location", "ped_movement")
PED_SOURCE_COLUMNS: dict[str, str] = {
    "age_of_casualty": "ped_age",
    "pedestrian_location": "ped_location",
    "pedestrian_movement": "ped_movement",
}
VEHICLE_NUMERIC_STATS: tuple[str, ...] = ("min", "mean", "max")


@dataclass(frozen=True)
class ColumnSpec:
    """Decision record for one raw column.

    Attributes:
        table: Source table (``collision``, ``vehicle`` or ``casualty``).
        name: Raw column name.
        decision: One of :data:`DECISIONS`.
        reason: One-line justification.
        dtype: pandas dtype used when reading the CSV (``Int32``, ``float64`` or ``string``).
        kind: ``categorical`` (integer code), ``numeric``, ``identifier``, ``temporal`` or
            ``text`` (string category).
    """

    table: str
    name: str
    decision: str
    reason: str
    dtype: str = "Int32"
    kind: str = "categorical"


def _spec(
    table: str,
    name: str,
    decision: str,
    reason: str,
    dtype: str = "Int32",
    kind: str = "categorical",
) -> ColumnSpec:
    """Build a :class:`ColumnSpec` (short alias keeping the registry readable)."""
    return ColumnSpec(table, name, decision, reason, dtype, kind)


_ID = ("Int32", "identifier")
_STR_ID = ("string", "identifier")
_NUM_INT = ("Int32", "numeric")
_NUM_FLOAT = ("float64", "numeric")
_TEXT = ("string", "text")
_TEMPORAL = ("string", "temporal")

_COLLISION: tuple[ColumnSpec, ...] = (
    _spec("collision", "collision_index", "drop_id", "Unique collision id; join key.", *_STR_ID),
    _spec("collision", "collision_year", "drop_id", "Split key (year).", *_ID),
    _spec("collision", "collision_ref_no", "drop_id", "In-year police reference number.", *_STR_ID),
    _spec(
        "collision", "location_easting_osgr", "drop_other", "Duplicate of longitude.", *_NUM_FLOAT
    ),
    _spec(
        "collision", "location_northing_osgr", "drop_other", "Duplicate of latitude.", *_NUM_FLOAT
    ),
    _spec("collision", "longitude", "keep", "Location known at report time.", *_NUM_FLOAT),
    _spec("collision", "latitude", "keep", "Location known at report time.", *_NUM_FLOAT),
    _spec("collision", "police_force", "keep", "Reporting force; known at report time."),
    _spec(
        "collision", "collision_severity", "target", "Target: Fatal/Serious (1,2) vs Slight (3)."
    ),
    _spec(
        "collision", "number_of_vehicles", "keep", "Vehicles involved; pre-impact fact.", *_NUM_INT
    ),
    _spec("collision", "number_of_casualties", "drop_leakage", "Injury-outcome count.", *_NUM_INT),
    _spec(
        "collision",
        "date",
        "keep",
        "Parsed to month / day_of_year (iso_week = split key).",
        *_TEMPORAL,
    ),
    _spec("collision", "day_of_week", "keep", "Temporal context."),
    _spec("collision", "time", "keep", "Parsed to hour.", *_TEMPORAL),
    _spec("collision", "local_authority_district", "drop_other", "100% missing (-1).", *_ID),
    _spec(
        "collision",
        "local_authority_ons_district",
        "keep",
        "Geography (frequency/target encoded downstream).",
        *_TEXT,
    ),
    _spec(
        "collision", "local_authority_highway", "drop_other", "Redundant with ONS district.", *_TEXT
    ),
    _spec(
        "collision",
        "local_authority_highway_current",
        "drop_other",
        "Redundant with ONS district.",
        *_TEXT,
    ),
    _spec("collision", "first_road_class", "keep", "Road context."),
    _spec(
        "collision", "first_road_number", "drop_other", "Road identifier (high cardinality).", *_ID
    ),
    _spec("collision", "road_type", "keep", "Road context."),
    _spec("collision", "speed_limit", "keep", "Road context (valid 20-70 mph).", *_NUM_INT),
    _spec(
        "collision",
        "junction_detail_historic",
        "drop_other",
        "Historic coding; superseded by junction_detail.",
    ),
    _spec("collision", "junction_detail", "keep", "Road context."),
    _spec(
        "collision",
        "junction_control",
        "keep",
        "Road context; structural missing when not at junction -> 'not at junction'.",
    ),
    _spec("collision", "second_road_class", "keep", "Road context."),
    _spec(
        "collision", "second_road_number", "drop_other", "Road identifier (high cardinality).", *_ID
    ),
    _spec(
        "collision",
        "pedestrian_crossing_human_control_historic",
        "drop_other",
        "Historic coding; superseded by pedestrian_crossing.",
    ),
    _spec(
        "collision",
        "pedestrian_crossing_physical_facilities_historic",
        "drop_other",
        "Historic coding; superseded by pedestrian_crossing.",
    ),
    _spec("collision", "pedestrian_crossing", "keep", "Road context."),
    _spec("collision", "light_conditions", "keep", "Environment."),
    _spec("collision", "weather_conditions", "keep", "Environment."),
    _spec("collision", "road_surface_conditions", "keep", "Environment."),
    _spec(
        "collision",
        "special_conditions_at_site",
        "drop_other",
        "Phased out (59% / 87% missing in 2024 / 2025).",
    ),
    _spec(
        "collision",
        "carriageway_hazards_historic",
        "drop_other",
        "Historic coding; superseded by carriageway_hazards.",
    ),
    _spec("collision", "carriageway_hazards", "keep", "Environment."),
    _spec("collision", "urban_or_rural_area", "keep", "Road context."),
    _spec(
        "collision",
        "did_police_officer_attend_scene_of_accident",
        "eval_only",
        "Attendance is decided by severity; kept outside X as evaluation stratifier.",
    ),
    _spec("collision", "trunk_road_flag", "keep", "Road context."),
    _spec(
        "collision",
        "lsoa_of_accident_location",
        "drop_other",
        "Too high cardinality (>30k levels).",
        *_TEXT,
    ),
    _spec("collision", "enhanced_severity_collision", "drop_leakage", "Target-derived severity."),
    _spec(
        "collision",
        "collision_injury_based",
        "keep",
        "Borderline: force reporting system fixed before crash; measurement artefact (S3).",
    ),
    _spec(
        "collision",
        "collision_adjusted_severity_serious",
        "drop_leakage",
        "Target-derived adjustment weight.",
        *_NUM_FLOAT,
    ),
    _spec(
        "collision",
        "collision_adjusted_severity_slight",
        "drop_leakage",
        "Target-derived adjustment weight.",
        *_NUM_FLOAT,
    ),
)

_VEHICLE: tuple[ColumnSpec, ...] = (
    _spec("vehicle", "collision_index", "drop_id", "Join key.", *_STR_ID),
    _spec("vehicle", "collision_year", "drop_id", "Split key (year).", *_ID),
    _spec("vehicle", "collision_ref_no", "drop_id", "In-year police reference number.", *_STR_ID),
    _spec("vehicle", "vehicle_reference", "drop_id", "Within-collision vehicle id (order).", *_ID),
    _spec("vehicle", "vehicle_type", "keep", "Who/what was involved."),
    _spec("vehicle", "towing_and_articulation", "keep", "Vehicle configuration pre-impact."),
    _spec(
        "vehicle",
        "vehicle_manoeuvre_historic",
        "drop_other",
        "Historic coding; superseded by vehicle_manoeuvre.",
    ),
    _spec("vehicle", "vehicle_manoeuvre", "keep", "Pre-impact manoeuvre."),
    _spec("vehicle", "vehicle_direction_from", "keep", "Pre-impact direction."),
    _spec("vehicle", "vehicle_direction_to", "keep", "Pre-impact direction."),
    _spec(
        "vehicle",
        "vehicle_location_restricted_lane_historic",
        "drop_other",
        "Historic coding; superseded by vehicle_location_restricted_lane.",
    ),
    _spec("vehicle", "vehicle_location_restricted_lane", "keep", "Pre-impact location."),
    _spec("vehicle", "junction_location", "keep", "Pre-impact location at junction."),
    _spec(
        "vehicle",
        "skidding_and_overturning",
        "tier_b",
        "Crash mechanics / impact consequence (sensitivity S1 only).",
    ),
    _spec(
        "vehicle",
        "hit_object_in_carriageway",
        "tier_b",
        "Crash mechanics / impact consequence (sensitivity S1 only).",
    ),
    _spec(
        "vehicle",
        "vehicle_leaving_carriageway",
        "tier_b",
        "Crash mechanics / impact consequence (sensitivity S1 only).",
    ),
    _spec(
        "vehicle",
        "hit_object_off_carriageway",
        "tier_b",
        "Crash mechanics / impact consequence (sensitivity S1 only).",
    ),
    _spec(
        "vehicle",
        "first_point_of_impact",
        "tier_b",
        "Crash mechanics / impact consequence (sensitivity S1 only).",
    ),
    _spec("vehicle", "vehicle_left_hand_drive", "keep", "Vehicle attribute."),
    _spec(
        "vehicle",
        "journey_purpose_of_driver_historic",
        "drop_other",
        "Historic coding; superseded by journey_purpose_of_driver.",
    ),
    _spec("vehicle", "journey_purpose_of_driver", "keep", "Driver context."),
    _spec("vehicle", "sex_of_driver", "keep", "Driver attribute."),
    _spec("vehicle", "age_of_driver", "keep", "Driver attribute (min/mean/max).", *_NUM_INT),
    _spec("vehicle", "age_band_of_driver", "drop_other", "Redundant with age_of_driver."),
    _spec("vehicle", "engine_capacity_cc", "keep", "Vehicle attribute (min/mean/max).", *_NUM_INT),
    _spec("vehicle", "propulsion_code", "keep", "Vehicle attribute."),
    _spec("vehicle", "age_of_vehicle", "drop_other", "100% missing in 2025.", *_NUM_INT),
    _spec(
        "vehicle", "generic_make_model", "drop_other", "High cardinality; largely redacted.", *_TEXT
    ),
    _spec("vehicle", "driver_imd_decile", "keep", "Driver home-area deprivation."),
    _spec("vehicle", "lsoa_of_driver", "drop_other", "Too high cardinality.", *_TEXT),
    _spec("vehicle", "escooter_flag", "keep", "Vehicle attribute (e-scooter)."),
    _spec("vehicle", "driver_distance_banding", "drop_other", "100% missing in 2023."),
)

_CASUALTY: tuple[ColumnSpec, ...] = (
    _spec("casualty", "collision_index", "drop_id", "Join key.", *_STR_ID),
    _spec("casualty", "collision_year", "drop_id", "Split key (year).", *_ID),
    _spec("casualty", "collision_ref_no", "drop_id", "In-year police reference number.", *_STR_ID),
    _spec("casualty", "vehicle_reference", "drop_id", "Link to vehicle row.", *_ID),
    _spec(
        "casualty",
        "casualty_reference",
        "drop_id",
        "Casualty id; used only to pick the first pedestrian.",
        *_ID,
    ),
    _spec("casualty", "casualty_class", "keep", "Used only for ped_involved (class == 3)."),
    _spec(
        "casualty",
        "sex_of_casualty",
        "drop_leakage",
        "Injury composition (rows exist only for injured).",
    ),
    _spec("casualty", "age_of_casualty", "keep", "First pedestrian only -> ped_age.", *_NUM_INT),
    _spec("casualty", "age_band_of_casualty", "drop_other", "Redundant with age_of_casualty."),
    _spec("casualty", "casualty_severity", "drop_leakage", "Target-derived."),
    _spec("casualty", "pedestrian_location", "keep", "First pedestrian only -> ped_location."),
    _spec("casualty", "pedestrian_movement", "keep", "First pedestrian only -> ped_movement."),
    _spec("casualty", "car_passenger", "drop_leakage", "Injury composition."),
    _spec("casualty", "bus_or_coach_passenger", "drop_leakage", "Injury composition."),
    _spec("casualty", "pedestrian_road_maintenance_worker", "drop_leakage", "Injury composition."),
    _spec("casualty", "casualty_type", "drop_leakage", "Injury composition."),
    _spec("casualty", "casualty_imd_decile", "drop_leakage", "Injury composition."),
    _spec("casualty", "lsoa_of_casualty", "drop_other", "Too high cardinality.", *_TEXT),
    _spec("casualty", "enhanced_casualty_severity", "drop_leakage", "Target-derived."),
    _spec(
        "casualty", "casualty_injury_based", "drop_other", "Duplicate of collision_injury_based."
    ),
    _spec(
        "casualty",
        "casualty_adjusted_severity_serious",
        "drop_leakage",
        "Target-derived adjustment weight.",
        *_NUM_FLOAT,
    ),
    _spec(
        "casualty",
        "casualty_adjusted_severity_slight",
        "drop_leakage",
        "Target-derived adjustment weight.",
        *_NUM_FLOAT,
    ),
    _spec("casualty", "casualty_distance_banding", "drop_other", "100% missing in 2023."),
    _spec("casualty", "escooter_flag", "drop_other", "Casualty field introduced only in 2025."),
)

REGISTRY: dict[str, tuple[ColumnSpec, ...]] = {
    "collision": _COLLISION,
    "vehicle": _VEHICLE,
    "casualty": _CASUALTY,
}


def _check_table(table: str) -> None:
    """Raise if ``table`` is not a known table name."""
    if table not in REGISTRY:
        raise KeyError(f"Unknown table {table!r}; expected one of {TABLES}")


def table_specs(table: str) -> tuple[ColumnSpec, ...]:
    """Return the column specs of one table in raw CSV order.

    Args:
        table: Table name.

    Returns:
        Tuple of :class:`ColumnSpec`.
    """
    _check_table(table)
    return REGISTRY[table]


def get_spec(table: str, name: str) -> ColumnSpec:
    """Return the spec of a single column.

    Args:
        table: Table name.
        name: Raw column name.

    Returns:
        The matching :class:`ColumnSpec`.

    Raises:
        KeyError: If the column is not registered.
    """
    for spec in table_specs(table):
        if spec.name == name:
            return spec
    raise KeyError(f"Column {name!r} not registered for table {table!r}")


def raw_columns(table: str) -> list[str]:
    """Return all registered raw column names of a table."""
    return [s.name for s in table_specs(table)]


def columns_by_decision(table: str, *decisions: str) -> list[str]:
    """Return the raw columns of ``table`` whose decision is in ``decisions``.

    Args:
        table: Table name.
        *decisions: One or more decisions from :data:`DECISIONS`.

    Returns:
        Column names in raw order.
    """
    unknown = set(decisions) - set(DECISIONS)
    if unknown:
        raise ValueError(f"Unknown decisions: {sorted(unknown)}")
    return [s.name for s in table_specs(table) if s.decision in decisions]


def kept_columns(table: str) -> list[str]:
    """Return the raw columns of ``table`` with decision ``keep``."""
    return columns_by_decision(table, "keep")


def tier_b_columns(table: str = "vehicle") -> list[str]:
    """Return the Tier-B (borderline crash mechanics) columns of ``table``."""
    return columns_by_decision(table, "tier_b")


def read_dtypes(table: str) -> dict[str, str]:
    """Return the explicit ``{column: dtype}`` mapping used to read a raw CSV."""
    return {s.name: s.dtype for s in table_specs(table)}


def deny_listed_columns(table: str | None = None) -> set[str]:
    """Return raw column names that must never be default model features.

    Args:
        table: Restrict to one table; ``None`` unions all three tables. A raw name that is kept in
            one table but dropped in another (``escooter_flag``) only appears for the dropping
            table, so prefer passing ``table`` when checking raw columns.

    Returns:
        Set of deny-listed raw column names.
    """
    tables = TABLES if table is None else (table,)
    return {s.name for t in tables for s in table_specs(t) if s.decision in NON_FEATURE_DECISIONS}


def processed_deny_list(columns: Iterable[str]) -> set[str]:
    """Return the processed (collision-level) columns that are excluded from default features.

    Excludes metadata/split keys, the targets, the eval-only stratifier, every ``tierb_*`` column
    and any raw collision column that is not a ``keep`` column.

    Args:
        columns: Column names of the processed collision table.

    Returns:
        Set of excluded column names present in ``columns``.
    """
    cols = list(columns)
    fixed = {TARGET_COL, SEVERITY3_COL, POLICE_ATTENDED_COL, *META_COLUMNS}
    fixed |= deny_listed_columns("collision")
    return {c for c in cols if c in fixed or c.startswith(TIERB_PREFIX)}


def default_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return the default (F0-eligible) feature columns among processed columns.

    Allowed: kept non-temporal collision columns, derived temporal features, vehicle aggregates
    (``n_vehicle_rows`` and ``veh_*``) and pedestrian fields. Everything in
    :func:`processed_deny_list` is excluded.

    Args:
        columns: Column names of the processed collision table.

    Returns:
        Feature column names in the order given.
    """
    cols = list(columns)
    denied = processed_deny_list(cols)
    collision_feats = {
        s.name for s in table_specs("collision") if s.decision == "keep" and s.kind != "temporal"
    }
    allowed_exact = collision_feats | set(DERIVED_TEMPORAL_FEATURES) | set(PED_FEATURES)
    allowed_exact.add(N_VEHICLE_ROWS_COL)
    return [
        c for c in cols if c not in denied and (c in allowed_exact or c.startswith(VEHICLE_PREFIX))
    ]


def leakage_audit_frame() -> pd.DataFrame:
    """Render the full column registry as a DataFrame (one row per raw column).

    Returns:
        DataFrame with columns ``table, name, decision, reason, dtype, kind,
        feeds_default_features`` (whether the column, directly or via a derived/aggregated
        feature, feeds the default feature set).
    """
    rows = [asdict(s) for t in TABLES for s in table_specs(t)]
    frame = pd.DataFrame(rows)
    frame["feeds_default_features"] = frame["decision"].eq("keep")
    return frame
