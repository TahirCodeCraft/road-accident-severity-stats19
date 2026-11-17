"""Global random-seed control for reproducible experiments."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, ``PYTHONHASHSEED`` and (if installed) PyTorch.

    Note that ``PYTHONHASHSEED`` only affects interpreters started after it is set (e.g. worker
    subprocesses); the current process's string hashing is fixed at start-up.

    Args:
        seed: Non-negative integer seed.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch_if_available(seed)
    logger.debug(f"Global seed set to {seed}")


def _seed_torch_if_available(seed: int) -> None:
    """Seed PyTorch CPU and CUDA generators when torch is importable.

    Args:
        seed: Seed value.
    """
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
