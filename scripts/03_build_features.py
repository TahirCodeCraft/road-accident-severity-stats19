"""Build the F0 and F1full feature parquets for both phases (tune, refit).

Outputs ``data/processed/features/{F0,F1full}_{tune,refit}.parquet`` (see
``src/features/frame.py`` for the contract).

Example:
    .venv/bin/python scripts/03_build_features.py --log-level INFO
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

from src.features.build import BuildFeaturesConfig, build_all
from src.paths import REPORTS_DIR
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--phases", nargs="+", default=["tune", "refit"],
                        choices=["tune", "refit"], help="Phases to build.")
    return parser.parse_args()


def main() -> None:
    """Build and write all feature frames."""
    args = parse_args()
    setup_logging(args.log_level, log_file=REPORTS_DIR / "logs" / "03_build_features.log")
    config = BuildFeaturesConfig(phases=tuple(args.phases))
    set_seed(config.engineered.seed)
    logger.info(f"Feature build config: {config}")
    t0 = time.time()
    written = build_all(config)
    for (name, phase), path in sorted(written.items()):
        logger.info(f"{name:7s} {phase:5s} -> {path}")
    logger.info(f"Feature build finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
