"""Orchestration of the F0 / F1full feature build for both phases (``tune`` and ``refit``).

``tune``  phase: encoders fitted on FIT rows; rows of FIT / TUNE / CAL.
``refit`` phase: encoders fitted on FIT+TUNE rows; rows of FIT / TUNE / CAL / TEST.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.features.engineered import (
    EngineeredConfig,
    build_f1full,
    load_bank_holidays,
    vehicle_mix_features,
)
from src.features.frame import (
    FEATURES_DIR,
    PHASES,
    assemble,
    load_collisions,
    phase_rows,
    train_mask,
    write_frame,
)
from src.features.raw import build_f0, f0_vocab
from src.paths import COLLISIONS_PATH, VEHICLES_CLEAN_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildFeaturesConfig:
    """Paths and settings of the feature build.

    Attributes:
        collisions_path: Processed collision parquet.
        vehicles_path: Clean vehicle-level parquet.
        features_dir: Output directory of the feature parquets.
        phases: Phases to build.
        engineered: Engineered-feature configuration.
    """

    collisions_path: Path = COLLISIONS_PATH
    vehicles_path: Path = VEHICLES_CLEAN_PATH
    features_dir: Path = FEATURES_DIR
    phases: tuple[str, ...] = PHASES
    engineered: EngineeredConfig = field(default_factory=EngineeredConfig)


def build_phase_frames(collisions: pd.DataFrame, mix: pd.DataFrame, phase: str,
                       config: EngineeredConfig) -> dict[str, pd.DataFrame]:
    """Build the assembled F0 and F1full frames of one phase.

    Args:
        collisions: Full processed collision frame with ``split``.
        mix: Vehicle-mix features (label-free, all collisions).
        phase: ``tune`` or ``refit``.
        config: Engineered-feature configuration.

    Returns:
        ``{"F0": frame, "F1full": frame}``.
    """
    vocab = f0_vocab(collisions)
    rows = phase_rows(collisions, phase)
    train = train_mask(rows, phase)
    f0 = build_f0(rows, train, vocab)
    holidays = load_bank_holidays(config.bank_holidays_path)
    f1full = build_f1full(rows, f0, mix, train, config, holidays)
    return {"F0": assemble(rows, f0), "F1full": assemble(rows, f1full)}


def build_all(config: BuildFeaturesConfig) -> dict[tuple[str, str], Path]:
    """Build and write F0 / F1full for every configured phase.

    Args:
        config: Build configuration.

    Returns:
        ``{(feature_set, phase): parquet path}``.
    """
    collisions = load_collisions(config.collisions_path)
    t0 = time.time()
    mix = vehicle_mix_features(pd.read_parquet(config.vehicles_path))
    logger.info(f"Vehicle aggregation took {time.time() - t0:.1f}s")
    written: dict[tuple[str, str], Path] = {}
    for phase in config.phases:
        t0 = time.time()
        frames = build_phase_frames(collisions, mix, phase, config.engineered)
        for name, frame in frames.items():
            written[(name, phase)] = write_frame(frame, name, phase, config.features_dir)
        logger.info(f"Phase {phase} built in {time.time() - t0:.1f}s")
    return written
