"""Fit calibrators and choose thresholds on CAL for every completed cell/seed.

Writes per cell ``frozen_seed{s}.json`` + calibrators, and the long table
``reports/tables/validation_results.csv`` (one row per cell x seed x variant x operating point).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from pathlib import Path as _Path

import pandas as pd

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.posthoc import save_decision, validate_on_cal
from src.paths import EXPERIMENTS_DIR, TABLES_DIR
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def validate_cell(cell_dir: Path) -> list[dict]:
    """Validate all seeds of one cell.

    Args:
        cell_dir: Cell directory containing ``preds_cal_seed*.parquet``.

    Returns:
        Validation metric rows tagged with cell metadata.
    """
    spec = json.loads((cell_dir / "config.json").read_text())
    rows: list[dict] = []
    for pred_file in sorted(cell_dir.glob("preds_cal_seed*.parquet")):
        seed = int(pred_file.stem.split("seed")[-1])
        preds = pd.read_parquet(pred_file)
        decision, cals, cal_rows = validate_on_cal(preds["target"].to_numpy(),
                                                   preds["prob"].to_numpy())
        save_decision(cell_dir, seed, decision, cals)
        meta = {"cell_id": cell_dir.name, "model": spec["model"], "feature_set":
                spec["feature_set"], "cs": spec["cs"], "tag": spec.get("tag", ""), "seed": seed,
                "best_calibrator": decision.best_calibrator}
        rows.extend(meta | r for r in cal_rows)
    return rows


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-dir", default=str(EXPERIMENTS_DIR / "cells"))
    args = parser.parse_args()
    setup_logging()
    rows: list[dict] = []
    for cell_dir in sorted(p for p in Path(args.cells_dir).iterdir()):
        if not (cell_dir / "summary.json").exists():
            logger.warning("Skipping incomplete cell %s", cell_dir.name)
            continue
        rows.extend(validate_cell(cell_dir))
        logger.info("Validated %s", cell_dir.name)
    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "validation_results.csv", index=False)
    logger.info("Wrote %d validation rows for %d cells", len(df), df["cell_id"].nunique())


if __name__ == "__main__":
    main()
