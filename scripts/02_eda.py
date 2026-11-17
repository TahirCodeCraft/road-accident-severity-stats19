"""Exploratory data analysis: figures in reports/figures/eda_*.png and reports/tables/eda_*.csv.

Label-based views use 2023-2024 only (2025 is the sealed test year). 2025 appears only in
label-free covariate views (injury-based reporting share, missingness / schema drift).

Example:
    .venv/bin/python scripts/02_eda.py --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.codebook import Codebook, load_codebook
from src.evaluation import eda
from src.evaluation.eda import FactorSpec
from src.paths import (
    COLLISIONS_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    raw_csv_path,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ROAD_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec("speed_limit", "Speed limit", ordered=True, fallback_fmt="{} mph"),
    FactorSpec("urban_or_rural_area", "Urban / rural"),
    FactorSpec("road_type", "Road type"),
    FactorSpec("junction_detail", "Junction detail"),
)
ENVIRONMENT_FACTORS: tuple[FactorSpec, ...] = (
    FactorSpec("light_conditions", "Light conditions"),
    FactorSpec("weather_conditions", "Weather"),
    FactorSpec("road_surface_conditions", "Road surface"),
)

# (table, raw column, display label, block) for the schema-drift missingness view.
MISSINGNESS_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("collision", "road_type", "Road type", "Kept features"),
    ("collision", "junction_detail", "Junction detail", "Kept features"),
    ("collision", "weather_conditions", "Weather", "Kept features"),
    ("collision", "road_surface_conditions", "Road surface", "Kept features"),
    ("collision", "pedestrian_crossing", "Pedestrian crossing", "Kept features"),
    ("collision", "carriageway_hazards", "Carriageway hazards", "Kept features"),
    ("collision", "trunk_road_flag", "Trunk road flag", "Kept features"),
    ("vehicle", "age_of_driver", "Driver age (vehicle rows)", "Kept features"),
    ("vehicle", "engine_capacity_cc", "Engine capacity (vehicle rows)", "Kept features"),
    ("vehicle", "driver_imd_decile", "Driver IMD decile (vehicle rows)", "Kept features"),
    ("collision", "special_conditions_at_site", "Special conditions at site", "Dropped (drift)"),
    ("collision", "junction_detail_historic", "Junction detail (historic)", "Dropped (drift)"),
    ("collision", "carriageway_hazards_historic", "Carriageway hazards (historic)",
     "Dropped (drift)"),
    ("collision", "pedestrian_crossing_physical_facilities_historic",
     "Ped. crossing facilities (historic)", "Dropped (drift)"),
    ("vehicle", "vehicle_manoeuvre_historic", "Vehicle manoeuvre (historic)", "Dropped (drift)"),
    ("vehicle", "journey_purpose_of_driver_historic", "Journey purpose (historic)",
     "Dropped (drift)"),
    ("vehicle", "age_of_vehicle", "Age of vehicle (vehicle rows)", "Dropped (drift)"),
    ("vehicle", "driver_distance_banding", "Driver distance banding", "Dropped (drift)"),
    ("casualty", "casualty_distance_banding", "Casualty distance banding", "Dropped (drift)"),
    ("collision", "local_authority_district", "Local authority district", "Dropped (drift)"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--heatmap-min-n", type=int, default=100,
                        help="Minimum collisions per hour x day cell (default: 100).")
    parser.add_argument("--hex-min-n", type=int, default=40,
                        help="Minimum collisions per map hexagon (default: 40).")
    return parser.parse_args()


def raw_missingness(codebook: Codebook) -> pd.DataFrame:
    """Compute per-year missing shares for :data:`MISSINGNESS_FIELDS` from the raw CSVs.

    Missing = empty cell, ``-1`` or a field-specific "unknown" code from the data guide. This view
    never reads the severity label.

    Args:
        codebook: Parsed data guide.

    Returns:
        Frame indexed by display label with one column per year plus ``group``.
    """
    columns: dict[int, pd.Series] = {}
    for year in eda.ALL_YEARS:
        parts = []
        for table in ("collision", "vehicle", "casualty"):
            fields = [f for t, f, _, _ in MISSINGNESS_FIELDS if t == table]
            path = raw_csv_path(table, year)
            header = pd.read_csv(path, nrows=0).columns
            frame = pd.read_csv(path, usecols=[f for f in fields if f in header], low_memory=False)
            codes = {f: codebook.missing_for(table, f) for f in fields}
            share = eda.missing_share(frame, fields, codes)
            share.index = [f"{table}.{f}" for f in share.index]
            parts.append(share)
            logger.info(f"Missingness {table} {year}: {len(frame):,} rows")
        columns[year] = pd.concat(parts)
    result = pd.DataFrame(columns)
    keys = [f"{t}.{f}" for t, f, _, _ in MISSINGNESS_FIELDS]
    result = result.loc[keys]
    result.index = [label for _, _, label, _ in MISSINGNESS_FIELDS]
    result["group"] = [group for *_, group in MISSINGNESS_FIELDS]
    return result


def run(args: argparse.Namespace) -> None:
    """Run every EDA step and write figures and tables.

    Args:
        args: Parsed CLI arguments.
    """
    eda.set_style()
    codebook = load_codebook()
    df = pd.read_parquet(COLLISIONS_PATH)
    logger.info(f"Loaded {len(df):,} collisions from {COLLISIONS_PATH}")
    dev = eda.dev_rows(df)
    logger.info(f"Label-based EDA on {len(dev):,} rows ({eda.DEV_YEARS}); test year excluded")
    collector = eda.SummaryCollector()
    fig = FIGURES_DIR

    eda.plot_class_balance(dev, fig / "eda_01_class_balance.png", collector)
    eda.plot_monthly_trend(df, fig / "eda_02_monthly_trend_reporting.png", collector)
    eda.plot_hour_dow_heatmap(dev, fig / "eda_03_hour_dow_heatmap.png", collector, codebook,
                              min_n=args.heatmap_min_n)
    eda.plot_factor_panels(dev, ROAD_FACTORS, fig / "eda_04_road_factors.png", collector,
                           codebook, "Speed limit and rural roads separate severity most",
                           section="road")
    eda.plot_factor_panels(dev, ENVIRONMENT_FACTORS, fig / "eda_05_environment_factors.png",
                           collector, codebook,
                           "Darkness without street lighting matters; rain and wet roads barely do",
                           section="env",
                           ncols=2)
    eda.plot_vehicle_mix(dev, fig / "eda_06_vehicle_mix.png", collector)
    eda.plot_hexbin_map(dev, fig / "eda_07_geo_hexbin.png", collector, min_count=args.hex_min_n)
    eda.plot_police_force_rank(dev, fig / "eda_08_police_force_rank.png", collector, codebook)
    eda.plot_reporting_system(dev, fig / "eda_09_reporting_system.png", collector, codebook)
    eda.plot_police_attendance(dev, fig / "eda_10_police_attendance.png", collector, codebook)

    missing = raw_missingness(codebook)
    eda.plot_missingness(missing, fig / "eda_11_missingness_drift.png")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    missing_path = TABLES_DIR / "eda_missingness.csv"
    missing.rename_axis("field").reset_index().to_csv(missing_path, index=False)
    logger.info(f"Wrote {missing_path}")

    summary_path = TABLES_DIR / "eda_summary.csv"
    collector.to_frame().to_csv(summary_path, index=False)
    logger.info(f"Wrote {summary_path} ({len(collector.rows)} rows)")


def main() -> None:
    """Entry point."""
    args = parse_args()
    setup_logging(args.log_level, log_file=REPORTS_DIR / "logs" / "02_eda.log")
    run(args)


if __name__ == "__main__":
    main()
