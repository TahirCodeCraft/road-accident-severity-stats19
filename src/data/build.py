"""Build the collision-level modelling table and the clean vehicle-level table."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.aggregate import (
    VehicleAggConfig,
    aggregate_vehicles,
    attach_aggregates,
    pedestrian_features,
)
from src.data.clean import (
    ValidityReport,
    clean_casualties,
    clean_collisions,
    clean_vehicles,
    collision_feature_columns,
)
from src.data.codebook import Codebook, load_codebook, log_missing_codes, missing_code_summary
from src.data.load import DEFAULT_YEARS, read_all
from src.data.schema import (
    POLICE_ATTENDED_COL,
    TARGET_COL,
    columns_by_decision,
    default_feature_columns,
    kept_columns,
    leakage_audit_frame,
    tier_b_columns,
)
from src.data.split import split_summary
from src.paths import DATA_GUIDE_PATH, PROCESSED_DIR, RAW_DIR, TABLES_DIR

logger = logging.getLogger(__name__)
TEST_YEAR_STR = "2025"

DQ_KEY_FIELDS: tuple[str, ...] = (
    "hour",
    "longitude",
    "local_authority_ons_district",
    "speed_limit",
    "road_type",
    "first_road_class",
    "second_road_class",
    "junction_detail",
    "junction_control",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "carriageway_hazards",
    "trunk_road_flag",
    POLICE_ATTENDED_COL,
    "veh_age_of_driver_mean",
    "veh_engine_capacity_cc_mean",
)


@dataclass(frozen=True)
class BuildConfig:
    """Configuration of the dataset build.

    Attributes:
        years: Data years to load.
        fit_year: Year defining vehicle code levels.
        raw_dir: Raw CSV directory.
        data_guide_path: Path of the data guide workbook.
        processed_dir: Output directory for parquet files.
        tables_dir: Output directory for report tables.
    """

    years: tuple[int, ...] = DEFAULT_YEARS
    fit_year: int = 2023
    raw_dir: Path = RAW_DIR
    data_guide_path: Path = DATA_GUIDE_PATH
    processed_dir: Path = PROCESSED_DIR
    tables_dir: Path = TABLES_DIR


@dataclass
class BuildOutputs:
    """Artefacts produced by :func:`build_dataset`.

    Attributes:
        collisions: One row per collision (all years).
        vehicles: Clean vehicle-level frame.
        validity: Collision validity report.
        codebook: Parsed data guide.
        tables: Report tables keyed by file stem.
    """

    collisions: pd.DataFrame
    vehicles: pd.DataFrame
    validity: ValidityReport
    codebook: Codebook
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def assert_collision_integrity(collisions: pd.DataFrame, n_raw: int) -> None:
    """Assert one row per raw collision and a unique ``collision_index``.

    Args:
        collisions: Final collision table.
        n_raw: Number of raw collision rows.

    Raises:
        AssertionError: If the checks fail.
    """
    assert len(collisions) == n_raw, f"row count {len(collisions)} != raw {n_raw}"
    assert collisions["collision_index"].is_unique, "collision_index not unique"


def _missing_pct(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Return ``{missing_pct_<col>: % missing}`` for columns present in ``frame``."""
    return {
        f"missing_pct_{c}": round(100 * float(frame[c].isna().mean()), 2)
        for c in columns
        if c in frame.columns
    }


def _quality_metrics(collisions: pd.DataFrame, vehicles: pd.DataFrame) -> dict[str, float]:
    """Compute quality metrics for one slice of the collision and vehicle tables."""
    metrics: dict[str, float] = {
        "rows": float(len(collisions)),
        "vehicle_rows": float(len(vehicles)),
        "severe_rate": round(float(collisions[TARGET_COL].mean()), 4),
        "injury_based_share": round(float(collisions["collision_injury_based"].mean()), 4),
        "ped_involved_share": round(float(collisions["ped_involved"].mean()), 4),
    }
    metrics |= _missing_pct(collisions, list(DQ_KEY_FIELDS))
    vehicle_fields = kept_columns("vehicle") + tier_b_columns("vehicle")
    vehicle_pct = _missing_pct(vehicles, vehicle_fields)
    metrics |= {k.replace("missing_pct_", "missing_pct_vehrow_"): v for k, v in vehicle_pct.items()}
    return metrics


def data_quality_table(collisions: pd.DataFrame, vehicles: pd.DataFrame) -> pd.DataFrame:
    """Build the data-quality table: one row per metric, one column per year plus ``all``.

    Args:
        collisions: Final collision table.
        vehicles: Clean vehicle-level frame.

    Returns:
        DataFrame indexed by metric name.
    """
    columns: dict[str, dict[str, float]] = {}
    for year, coll in collisions.groupby("collision_year"):
        veh = vehicles.loc[vehicles["collision_year"].eq(year)]
        columns[str(year)] = _quality_metrics(coll, veh)
    columns["all"] = _quality_metrics(collisions, vehicles)
    table = pd.DataFrame(columns)
    table.index.name = "metric"
    # Sealed test year: no label statistics before the freeze (only covariate quality).
    label_rows = [m for m in table.index if "severe" in m or "fatal" in m or "serious" in m]
    table.loc[label_rows, [c for c in table.columns if c in (TEST_YEAR_STR, "all")]] = float("nan")
    return table


def log_quality(table: pd.DataFrame) -> None:
    """Log the headline data-quality numbers per year."""
    for metric in ("rows", "severe_rate", "injury_based_share", "ped_involved_share"):
        logger.info(f"DQ {metric}: {table.loc[metric].to_dict()}")
    high = table.drop(index=["rows", "vehicle_rows"]).max(axis=1)
    high = high[high > 5].sort_values(ascending=False)
    logger.info(f"DQ fields with >5% missing in some year: {high.round(1).to_dict()}")


def build_dataset(config: BuildConfig | None = None) -> BuildOutputs:
    """Run the full build: load, clean, aggregate, join and validate.

    Args:
        config: Build configuration (defaults to :class:`BuildConfig`).

    Returns:
        :class:`BuildOutputs` with the collision and vehicle tables and report tables.
    """
    config = config or BuildConfig()
    codebook = load_codebook(config.data_guide_path)
    report_cols = {
        "collision": collision_feature_columns() + columns_by_decision("collision", "eval_only"),
        "vehicle": kept_columns("vehicle") + tier_b_columns("vehicle"),
        "casualty": kept_columns("casualty"),
    }
    log_missing_codes(codebook, report_cols)
    raw = read_all(config.years, config.raw_dir)

    collisions, validity = clean_collisions(raw["collision"], codebook)
    vehicles = clean_vehicles(raw["vehicle"], codebook)
    casualties = clean_casualties(raw["casualty"], codebook)

    vehicle_agg = aggregate_vehicles(vehicles, VehicleAggConfig(fit_year=config.fit_year))
    ped = pedestrian_features(casualties)
    collisions = attach_aggregates(collisions, vehicle_agg, ped)
    assert_collision_integrity(collisions, len(raw["collision"]))
    n_no_vehicle = int(collisions["n_vehicle_rows"].eq(0).sum())
    logger.info(f"Collisions without vehicle rows: {n_no_vehicle}")
    n_features = len(default_feature_columns(collisions.columns))
    logger.info(f"collisions: {collisions.shape}; default feature columns: {n_features}")

    tables = {
        "data_quality": data_quality_table(collisions, vehicles),
        "leakage_audit": leakage_audit_frame(),
        "split_summary": split_summary(collisions),
        "missing_code_map": missing_code_summary(codebook, report_cols),
    }
    log_quality(tables["data_quality"])
    return BuildOutputs(collisions, vehicles, validity, codebook, tables)


def write_outputs(outputs: BuildOutputs, config: BuildConfig | None = None) -> None:
    """Write parquet files and report tables.

    Args:
        outputs: Result of :func:`build_dataset`.
        config: Build configuration (output directories).
    """
    config = config or BuildConfig()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    collisions_path = config.processed_dir / "collisions.parquet"
    vehicles_path = config.processed_dir / "vehicles_clean.parquet"
    outputs.collisions.to_parquet(collisions_path, index=False)
    outputs.vehicles.to_parquet(vehicles_path, index=False)
    logger.info(f"Wrote {collisions_path} and {vehicles_path}")
    for stem, table in outputs.tables.items():
        path = config.tables_dir / f"{stem}.csv"
        keep_index = table.index.name is not None
        table.to_csv(path, index=keep_index)
        logger.info(f"Wrote {path}")
