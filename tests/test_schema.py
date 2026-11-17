"""Tests for the column registry / leakage audit."""

from __future__ import annotations

import csv

import pytest

from src.data.load import DEFAULT_YEARS
from src.data.schema import (
    DECISIONS,
    NON_FEATURE_DECISIONS,
    POLICE_ATTENDED_COL,
    SEVERITY3_COL,
    TABLES,
    TARGET_COL,
    columns_by_decision,
    default_feature_columns,
    deny_listed_columns,
    kept_columns,
    leakage_audit_frame,
    raw_columns,
    table_specs,
    tier_b_columns,
)
from src.paths import raw_csv_path

EXPECTED_TIER_B = {
    "skidding_and_overturning",
    "hit_object_in_carriageway",
    "vehicle_leaving_carriageway",
    "hit_object_off_carriageway",
    "first_point_of_impact",
}


def _csv_header(table: str, year: int) -> list[str]:
    """Read only the header line of a raw CSV."""
    with raw_csv_path(table, year).open(newline="") as handle:
        return next(csv.reader(handle))


@pytest.mark.parametrize("table", TABLES)
def test_registry_entries_are_valid(table: str) -> None:
    specs = table_specs(table)
    names = [s.name for s in specs]
    assert len(names) == len(set(names)), "duplicate column in registry"
    for spec in specs:
        assert spec.decision in DECISIONS
        assert spec.reason.strip()
        assert spec.dtype in {"Int32", "float64", "string"}


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("year", DEFAULT_YEARS)
def test_every_raw_column_classified(table: str, year: int) -> None:
    if not raw_csv_path(table, year).exists():
        pytest.skip("raw CSV not available")
    assert set(_csv_header(table, year)) == set(raw_columns(table))


def test_plan_decisions() -> None:
    assert set(tier_b_columns("vehicle")) == EXPECTED_TIER_B
    assert columns_by_decision("collision", "target") == ["collision_severity"]
    assert columns_by_decision("collision", "eval_only") == [
        "did_police_officer_attend_scene_of_accident"
    ]
    leak = set(columns_by_decision("collision", "drop_leakage"))
    assert {"number_of_casualties", "enhanced_severity_collision"} <= leak
    assert "collision_injury_based" in kept_columns("collision")
    assert set(kept_columns("casualty")) == {
        "casualty_class",
        "age_of_casualty",
        "pedestrian_location",
        "pedestrian_movement",
    }
    assert "casualty_severity" in deny_listed_columns("casualty")


def test_default_features_exclude_denied_columns() -> None:
    raw_names = {name for t in TABLES for name in raw_columns(t)}
    processed = [
        *sorted(raw_names),
        TARGET_COL,
        SEVERITY3_COL,
        POLICE_ATTENDED_COL,
        "iso_week",
        "split",
        "hour",
        "month",
        "day_of_year",
        "n_vehicle_rows",
        "veh_vehicle_type__9",
        "tierb_first_point_of_impact__1",
        "ped_involved",
        "ped_age",
    ]
    features = set(default_feature_columns(processed))
    denied = {
        s.name
        for t in TABLES
        for s in table_specs(t)
        if s.decision in {"drop_leakage", "drop_id", "eval_only", "tier_b", "target"}
    }
    assert not features & denied
    assert not features & {TARGET_COL, SEVERITY3_COL, POLICE_ATTENDED_COL, "iso_week", "split"}
    assert not any(f.startswith("tierb_") for f in features)
    assert {"date", "time"}.isdisjoint(features)
    assert {"hour", "veh_vehicle_type__9", "ped_involved", "speed_limit"} <= features
    # raw vehicle/casualty names never pass as-is (only via aggregates)
    assert "vehicle_type" not in features and "age_of_casualty" not in features


def test_leakage_audit_frame_covers_all_columns() -> None:
    audit = leakage_audit_frame()
    assert len(audit) == sum(len(raw_columns(t)) for t in TABLES)
    assert set(audit["decision"]) <= set(DECISIONS)
    non_feature = audit["decision"].isin(NON_FEATURE_DECISIONS)
    assert not audit.loc[non_feature, "feeds_default_features"].any()
