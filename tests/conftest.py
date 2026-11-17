"""Shared synthetic fixtures for data-pipeline tests (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.codebook import Codebook, build_labels, derive_missing_codes
from src.data.schema import get_spec, kept_columns, table_specs, tier_b_columns

SYNTHETIC_CODE_LIST = pd.DataFrame(
    [
        ("collision", "road_type", 6, "Single carriageway"),
        ("collision", "road_type", 9, "Unknown"),
        ("collision", "road_type", -1, "Data missing or out of range"),
        ("collision", "speed_limit", 99, "unknown (self reported)"),
        ("collision", "junction_detail", 0, "Not at or within 20 metres of junction"),
        ("collision", "junction_detail", 99, "unknown (self reported)"),
        ("collision", "junction_control", 9, "unknown (self reported)"),
        ("collision", "light_conditions", 1, "Daylight"),
        ("collision", "light_conditions", 7, "Darkness - lighting unknown"),
        ("collision", "weather_conditions", 9, "Unknown"),
        ("collision", "second_road_class", 9, "Unknown (self rep only)"),
        ("vehicle", "sex_of_driver", 1, "Male"),
        ("vehicle", "sex_of_driver", 3, "Not known"),
        ("vehicle", "vehicle_type", 9, "Car"),
        ("vehicle", "vehicle_type", 97, "Motorcycle - unknown cc"),
        ("vehicle", "vehicle_type", 99, "Unknown vehicle type (self rep only)"),
        ("vehicle", "propulsion_code", -1, "Undefined"),
        ("casualty", "pedestrian_location", 10, "Unknown or other"),
    ],
    columns=["table", "field", "code", "label"],
)


@pytest.fixture
def codebook() -> Codebook:
    """Codebook built from a small synthetic code list."""
    labels = build_labels(SYNTHETIC_CODE_LIST)
    return Codebook(labels=labels, missing_codes=derive_missing_codes(labels))


def _default_value(table: str, name: str) -> object:
    """Return a plausible default raw value for a registered column."""
    spec = get_spec(table, name)
    if spec.dtype == "string":
        return "E06000001"
    if spec.dtype == "float64":
        return 0.5
    return 1


def make_raw_collisions(n: int = 4) -> pd.DataFrame:
    """Return a raw-like collision frame with every registered column."""
    data = {s.name: [_default_value("collision", s.name)] * n for s in table_specs("collision")}
    frame = pd.DataFrame(data)
    for spec in table_specs("collision"):
        frame[spec.name] = frame[spec.name].astype(spec.dtype)
    frame["collision_index"] = pd.array([f"2023C{i:04d}" for i in range(n)], dtype="string")
    frame["collision_year"] = pd.array([2023] * n, dtype="Int32")
    frame["date"] = pd.array(["01/01/2023"] * n, dtype="string")
    frame["time"] = pd.array(["08:30"] * n, dtype="string")
    frame["speed_limit"] = pd.array([30] * n, dtype="Int32")
    frame["collision_severity"] = pd.array(([1, 2, 3, 3] * n)[:n], dtype="Int32")
    return frame


def make_vehicles(seed: int = 0, n_collisions: int = 30) -> pd.DataFrame:
    """Return a clean-like vehicle frame (kept + Tier-B fields) with random codes and NAs."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, size=n_collisions)
    idx = np.repeat([f"C{i:03d}" for i in range(n_collisions)], sizes)
    years = np.repeat(np.where(np.arange(n_collisions) < n_collisions // 2, 2023, 2024), sizes)
    refs = np.concatenate([np.arange(1, s + 1) for s in sizes])
    frame = pd.DataFrame(
        {
            "collision_index": pd.array(idx, dtype="string"),
            "collision_year": pd.array(years, dtype="Int32"),
            "vehicle_reference": pd.array(refs, dtype="Int32"),
        }
    )
    n = len(frame)
    for col in kept_columns("vehicle") + tier_b_columns("vehicle"):
        high = 80 if get_spec("vehicle", col).kind == "numeric" else 4
        values = pd.array(rng.integers(0, high, size=n), dtype="Int32")
        values[rng.random(n) < 0.15] = pd.NA
        frame[col] = values
    return frame
