"""Build data/processed/collisions.parquet and vehicles_clean.parquet plus report tables.

Example:
    .venv/bin/python scripts/01_build_dataset.py --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build import BuildConfig, build_dataset, write_outputs  # noqa: E402
from src.paths import REPORTS_DIR  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


def main() -> None:
    """Run the dataset build and write all outputs."""
    args = parse_args()
    setup_logging(args.log_level, log_file=REPORTS_DIR / "logs" / "01_build_dataset.log")
    config = BuildConfig()
    logger.info(f"Build config: {config}")
    outputs = build_dataset(config)
    write_outputs(outputs, config)
    logger.info(f"Split summary:\n{outputs.tables['split_summary'].to_string(index=False)}")


if __name__ == "__main__":
    main()
