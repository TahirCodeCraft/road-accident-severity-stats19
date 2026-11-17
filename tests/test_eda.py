"""Tests for the EDA helpers in src/evaluation/eda.py (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import eda


def test_wilson_ci_matches_reference() -> None:
    """Wilson interval for 10/100 at 95% matches the textbook value."""
    low, high = eda.wilson_ci(10, 100)
    assert low == pytest.approx(0.0552, abs=1e-4)
    assert high == pytest.approx(0.1744, abs=1e-4)


def test_wilson_ci_bounds_and_zero() -> None:
    """Interval stays inside [0, 1] for extreme counts and rejects n = 0."""
    low, high = eda.wilson_ci(np.array([0, 50]), np.array([50, 50]))
    assert low[0] == pytest.approx(0.0) and high[1] == pytest.approx(1.0)
    assert np.all(low >= 0) and np.all(high <= 1)
    with pytest.raises(ValueError):
        eda.wilson_ci(1, 0)


def test_dev_rows_excludes_and_refuses_test_year() -> None:
    """Label-based helpers never see 2025 rows."""
    df = pd.DataFrame({"collision_year": [2023, 2024, 2025], "target": [0, 1, 1]})
    out = eda.dev_rows(df)
    assert set(out["collision_year"]) == {2023, 2024}
    with pytest.raises(ValueError):
        eda.dev_rows(df, years=(2024, 2025))


def test_rate_table_counts_missing_level_and_min_n() -> None:
    """Missing values form their own group and small groups are dropped."""
    df = pd.DataFrame(
        {"g": pd.array([1, 1, 2, None, None, 3], dtype="Int32"), "target": [1, 0, 1, 0, 1, 1]}
    )
    table = eda.rate_table(df, "g", min_n=2)
    assert len(table) == 2
    missing = table.loc[table["g"].isna()].iloc[0]
    assert missing["n"] == 2 and missing["rate"] == pytest.approx(0.5)
    assert (table["ci_low"] <= table["rate"]).all() and (table["rate"] <= table["ci_high"]).all()


def test_label_levels_maps_codes_and_missing() -> None:
    """Codes map to labels, unknown codes use the fallback, NaN becomes the missing label."""
    values = pd.Series(pd.array([1, 30, None], dtype="Int32"))
    out = eda.label_levels(values, {1: "Daylight"}, fmt="{} mph")
    assert list(out) == ["Daylight", "30 mph", eda.MISSING_LABEL]


def test_vehicle_mix_flags() -> None:
    """Vehicle-type counts become involvement flags and vehicle-count bands."""
    df = pd.DataFrame(
        {
            "ped_involved": [1, 0, 0],
            "number_of_vehicles": [1, 2, 4],
            "veh_vehicle_type__1": [0, 1, 0],
            "veh_vehicle_type__3": [1, 0, 0],
            "veh_vehicle_type__21": [0, 0, 2],
        }
    )
    flags = eda.vehicle_mix_flags(df)
    assert list(flags["pedestrian"]) == ["Involved", "Not involved", "Not involved"]
    assert list(flags["motorcycle"]) == ["Involved", "Not involved", "Not involved"]
    assert list(flags["pedal_cycle"]) == ["Not involved", "Involved", "Not involved"]
    assert list(flags["hgv"]) == ["Not involved", "Not involved", "Involved"]
    assert list(flags["vehicles"]) == ["Single vehicle", "Two vehicles", "Three or more"]


def test_missing_share_uses_codes_and_absent_columns() -> None:
    """-1, field-specific unknown codes, NaN and absent columns all count as missing."""
    frame = pd.DataFrame({"a": [1, -1, 9, np.nan], "b": [1, 2, 3, 4]})
    share = eda.missing_share(frame, ["a", "b", "c"], {"a": frozenset({-1, 9})})
    assert share["a"] == pytest.approx(0.75)
    assert share["b"] == pytest.approx(0.0)
    assert share["c"] == pytest.approx(1.0)
