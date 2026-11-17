"""Project-wide logging configuration (console plus optional file handler)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = logging.INFO, log_file: Path | None = None) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Calling this repeatedly replaces previously installed handlers, so it is safe to call from
    every entry-point script.

    Args:
        level: Logging level (e.g. ``logging.INFO`` or ``"DEBUG"``).
        log_file: Optional path of a log file; parent directories are created as needed.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper kept for a single import location).

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        The logger instance.
    """
    return logging.getLogger(name)
