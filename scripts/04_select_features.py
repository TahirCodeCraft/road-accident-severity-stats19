"""Run F1 feature selection on FIT rows of F1full_tune and write the F1 feature frames.

Outputs:
    configs/features/f1_selected.yaml        frozen list (selected + dropped with reason)
    reports/tables/feature_selection.csv     per-feature diagnostics
    data/processed/features/F1_{tune,refit}.parquet  F1full restricted to the selected columns

Example:
    .venv/bin/python scripts/04_select_features.py --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow.parquet as pq

from src.features.frame import (
    ID_COLS,
    PHASES,
    feature_columns,
    feature_path,
    validate_dtypes,
    write_frame,
)
from src.features.selection import SelectionConfig, load_selected, select_features, write_selection
from src.paths import REPORTS_DIR
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

SOURCE_SET = "F1full"
TARGET_SET = "F1"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--n-repeats", type=int, default=5, help="Permutation repeats.")
    parser.add_argument("--n-jobs", type=int, default=12, help="LightGBM threads (<= 12).")
    parser.add_argument("--skip-selection", action="store_true",
                        help="Reuse configs/features/f1_selected.yaml; only write F1 frames.")
    return parser.parse_args()


def run_selection(config: SelectionConfig) -> list[str]:
    """Select features on FIT rows of F1full_tune and write the YAML / CSV outputs.

    Args:
        config: Selection configuration.

    Returns:
        Selected feature names.
    """
    path = feature_path(SOURCE_SET, "tune")
    frame = pd.read_parquet(path)
    fit = frame[frame["split"].eq("fit")].reset_index(drop=True)
    X, y = fit[feature_columns(fit)], fit["target"].to_numpy().astype(int)
    logger.info(f"Selection data: {path.name} FIT rows {X.shape[0]:,} x {X.shape[1]} features")
    result = select_features(X, y, config)
    write_selection(result, config, source=f"{path.name} (split == 'fit', 2023)")
    counts = result.table["stage"].replace("", "selected").value_counts().to_dict()
    logger.info(f"Selection outcome by stage: {counts}")
    return result.selected


def write_f1_frames(selected: list[str]) -> None:
    """Write F1_{phase} = F1full_{phase} restricted to the selected columns.

    Args:
        selected: Selected feature names.
    """
    for phase in PHASES:
        source = pd.read_parquet(feature_path(SOURCE_SET, phase))
        missing = [c for c in selected if c not in source.columns]
        if missing:
            raise KeyError(f"Selected features absent from {SOURCE_SET}_{phase}: {missing}")
        frame = source[list(ID_COLS) + selected]
        validate_dtypes(frame)
        write_frame(frame, TARGET_SET, phase)


def main() -> None:
    """Run selection (unless skipped) and write the F1 frames."""
    args = parse_args()
    setup_logging(args.log_level, log_file=REPORTS_DIR / "logs" / "04_select_features.log")
    config = SelectionConfig(n_repeats=args.n_repeats, n_jobs=min(args.n_jobs, 12))
    set_seed(config.seed)
    t0 = time.time()
    selected = load_selected() if args.skip_selection else run_selection(config)
    write_f1_frames(selected)
    n_source = len(pq.read_schema(feature_path(SOURCE_SET, "tune")).names) - len(ID_COLS)
    logger.info(f"Feature counts: {SOURCE_SET}={n_source}, {TARGET_SET}={len(selected)}")
    logger.info(f"Selection script finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
