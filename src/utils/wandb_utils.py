"""Fail-safe Weights & Biases helpers (offline by default, never crash the experiment)."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODE = "offline"


def resolve_mode(mode: str | None = None) -> str:
    """Resolve the wandb mode from the argument, ``WANDB_MODE`` env var, or ``"offline"``.

    Args:
        mode: Explicit mode (``"online"``, ``"offline"`` or ``"disabled"``) or ``None``.

    Returns:
        The resolved mode string.
    """
    if mode:
        return mode
    return os.environ.get("WANDB_MODE", DEFAULT_MODE)


def init_run(
    project: str,
    config: Mapping[str, Any] | None = None,
    name: str | None = None,
    group: str | None = None,
    mode: str | None = None,
    tags: list[str] | None = None,
    run_dir: Path | None = None,
) -> Any | None:
    """Start a wandb run, returning ``None`` (no-op) if wandb is unavailable or fails.

    Args:
        project: wandb project name.
        config: Full experiment configuration to attach to the run.
        name: Optional run name.
        group: Optional run group (e.g. the ablation cell).
        mode: wandb mode; defaults to ``WANDB_MODE`` or ``"offline"``.
        tags: Optional run tags.
        run_dir: Optional directory for local wandb files.

    Returns:
        The wandb ``Run`` object, or ``None`` if initialisation failed.
    """
    resolved = resolve_mode(mode)
    try:
        import wandb

        run = wandb.init(
            project=project,
            config=dict(config) if config is not None else None,
            name=name,
            group=group,
            mode=resolved,
            tags=tags,
            dir=str(run_dir) if run_dir is not None else None,
            reinit="finish_previous",
        )
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        logger.warning(f"wandb.init failed ({exc!r}); continuing without tracking")
        return None
    logger.info(f"wandb run started: project={project} name={name} mode={resolved}")
    return run


def log_metrics(run: Any | None, metrics: Mapping[str, Any], step: int | None = None) -> None:
    """Log a metrics dict to a run; silently no-op when ``run`` is ``None`` or logging fails.

    Args:
        run: Run returned by :func:`init_run` (may be ``None``).
        metrics: Mapping of metric name to value.
        step: Optional global step.
    """
    if run is None:
        return
    try:
        run.log(dict(metrics), step=step)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"wandb log failed ({exc!r})")


def log_summary(run: Any | None, summary: Mapping[str, Any]) -> None:
    """Write final values into the run summary; no-op when ``run`` is ``None``.

    Args:
        run: Run returned by :func:`init_run` (may be ``None``).
        summary: Mapping of summary keys to values.
    """
    if run is None:
        return
    try:
        run.summary.update(dict(summary))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"wandb summary update failed ({exc!r})")


def finish(run: Any | None) -> None:
    """Finish a run; no-op when ``run`` is ``None`` or finishing fails.

    Args:
        run: Run returned by :func:`init_run` (may be ``None``).
    """
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"wandb finish failed ({exc!r})")
