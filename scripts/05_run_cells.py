"""Run the ablation cells (HPO on FIT->TUNE, refit on FIT+TUNE, CAL predictions).

Examples:
    python scripts/05_run_cells.py run.models=[lgbm,rf]                 # local CPU
    python scripts/05_run_cells.py run.models=[xgb,cat] run.device=cuda  # GPU server
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path as _Path

import hydra
from omegaconf import DictConfig

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.training.runner import CellSpec, run_cell
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def build_cells(cfg: DictConfig) -> list[CellSpec]:
    """Expand the ablation config into a list of cells, applying run filters.

    Args:
        cfg: Hydra config.

    Returns:
        Cell specifications in execution order.
    """
    exp, run = cfg.experiment, cfg.run
    cells: list[CellSpec] = []
    for fs in exp.feature_sets:
        for cs in exp.cs_strategies:
            for m in exp.models:
                cells.append(CellSpec(m, fs, cs, n_trials=exp.n_trials[m]))
        for m in exp.smotenc_models:
            cells.append(CellSpec(m, fs, "smotenc", n_trials=exp.n_trials_smotenc))
    for m in exp.f1full_models:
        cells.append(CellSpec(m, "F1full", "none", n_trials=exp.n_trials[m]))
    keep = []
    for c in cells:
        if run.models and c.model not in run.models:
            continue
        if run.feature_sets and c.feature_set not in run.feature_sets:
            continue
        if run.cs and c.cs not in run.cs:
            continue
        c.seeds = list(exp.seeds)
        c.device = run.device if c.model in exp.gpu_models else "cpu"
        c.n_jobs = run.n_jobs
        keep.append(c)
    return keep


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point."""
    setup_logging()
    os.environ.setdefault("WANDB_MODE", cfg.wandb.mode)
    cells = build_cells(cfg)
    logger.info("Running %d cells: %s", len(cells), [c.cell_id for c in cells])
    if cfg.run.dry_run:
        return
    for c in cells:
        run_cell(c, overwrite=cfg.run.overwrite, wandb_project=cfg.wandb.project)


if __name__ == "__main__":
    main()
