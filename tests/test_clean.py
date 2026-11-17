"""Tests for codebook parsing and collision/vehicle cleaning."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import (
    check_speed_limits,
    clean_collisions,
    date_parts,
    fill_structural_junction,
    make_targets,
    map_missing_codes,
    parse_dates,
    parse_hour,
)
from src.data.codebook import Codebook, normalise_code
from tests.conftest import make_raw_collisions


def test_normalise_code() -> None:
    assert normalise_code(9) == 9
    assert normalise_code(99.0) == 99
    assert normalise_code("-1") == -1
    assert normalise_code("E06000001") == "E06000001"
    assert normalise_code(float("nan")) is None


def test_missing_codes_derivation(codebook: Codebook) -> None:
    assert codebook.missing_for("collision", "road_type") == {-1, 9}
    assert codebook.missing_for("collision", "speed_limit") == {-1, 99}
    assert codebook.missing_for("collision", "second_road_class") == {-1, 9}
    assert codebook.missing_for("vehicle", "sex_of_driver") == {-1, 3}
    assert codebook.missing_for("casualty", "pedestrian_location") == {-1, 10}
    # informative categories whose label merely contains "unknown" are kept
    assert codebook.missing_for("collision", "light_conditions") == {-1}
    assert codebook.missing_for("vehicle", "vehicle_type") == {-1, 99}
    # fields absent from the guide still treat -1 as missing
    assert codebook.missing_for("vehicle", "age_of_driver") == {-1}


def test_map_missing_codes_to_nan(codebook: Codebook) -> None:
    frame = pd.DataFrame(
        {
            "road_type": pd.array([6, 9, -1, 6], dtype="Int32"),
            "light_conditions": pd.array([1, 7, -1, 1], dtype="Int32"),
            "local_authority_ons_district": pd.array(["E1", "-1", "E2", "E3"], dtype="string"),
            "longitude": [-1.0, 0.5, 1.0, 2.0],
        }
    )
    out, counts = map_missing_codes(frame, "collision", codebook, list(frame.columns))
    assert out["road_type"].isna().tolist() == [False, True, True, False]
    assert out["light_conditions"].tolist()[:2] == [1, 7]
    assert out["light_conditions"].isna().tolist() == [False, False, True, False]
    assert out["local_authority_ons_district"].isna().tolist() == [False, True, False, False]
    assert out["longitude"].tolist() == [-1.0, 0.5, 1.0, 2.0]  # floats untouched
    assert counts["road_type"] == 2
    assert frame["road_type"].tolist() == [6, 9, -1, 6]  # input not mutated


def test_vehicle_unknown_codes(codebook: Codebook) -> None:
    frame = pd.DataFrame(
        {
            "sex_of_driver": pd.array([1, 3, -1], dtype="Int32"),
            "vehicle_type": pd.array([9, 97, 99], dtype="Int32"),
            "age_of_driver": pd.array([30, -1, 45], dtype="Int32"),
        }
    )
    out, _ = map_missing_codes(frame, "vehicle", codebook, list(frame.columns))
    assert out["sex_of_driver"].isna().tolist() == [False, True, True]
    assert out["vehicle_type"].isna().tolist() == [False, False, True]
    assert out["age_of_driver"].isna().tolist() == [False, True, False]


def test_structural_junction_fill() -> None:
    frame = pd.DataFrame(
        {
            "junction_detail": pd.array([0, 0, 13, -1], dtype="Int32"),
            "junction_control": pd.array([-1, 4, -1, -1], dtype="Int32"),
            "second_road_class": pd.array([-1, 0, 3, -1], dtype="Int32"),
        }
    )
    out, counts = fill_structural_junction(frame)
    assert out["junction_control"].tolist() == [0, 4, -1, -1]
    assert out["second_road_class"].tolist() == [0, 0, 3, -1]
    assert counts == {"junction_control": 1, "second_road_class": 1}


def test_target_mapping() -> None:
    sev = pd.Series([1, 2, 3, 3], dtype="Int32")
    out = make_targets(sev)
    assert out["target"].tolist() == [1, 1, 0, 0]
    assert out["severity3"].tolist() == [1, 2, 3, 3]
    with pytest.raises(ValueError):
        make_targets(pd.Series([1, 4], dtype="Int32"))
    with pytest.raises(ValueError):
        make_targets(pd.Series([1, None], dtype="Int32"))


def test_time_parsing() -> None:
    times = pd.Series(["00:00", "23:59", "7:05", "24:00", "12:61", "ab", None], dtype="string")
    hour = parse_hour(times)
    assert hour.tolist()[:3] == [0, 23, 7]
    assert hour.isna().tolist() == [False, False, False, True, True, True, True]


def test_date_parsing_and_iso_week() -> None:
    dates = parse_dates(pd.Series(["01/01/2023", "30/12/2024", "29/02/2024", "31/02/2024"]))
    assert dates.isna().tolist() == [False, False, False, True]
    parts = date_parts(dates.iloc[:3])
    assert parts["iso_week"].tolist() == [52, 1, 9]  # 1 Jan 2023 is ISO week 52 of 2022
    assert parts["month"].tolist() == [1, 12, 2]
    assert parts["day_of_year"].tolist() == [1, 365, 60]


def test_speed_limit_range() -> None:
    speed = pd.Series([20, 30, 15, None, 70, 99], dtype="Int32")
    out, n_bad = check_speed_limits(speed)
    assert n_bad == 2
    assert out.isna().tolist() == [False, False, True, True, False, True]


def test_clean_collisions_end_to_end(codebook: Codebook) -> None:
    raw = make_raw_collisions(4)
    raw["did_police_officer_attend_scene_of_accident"] = pd.array([1, 2, 3, -1], dtype="Int32")
    raw["road_type"] = pd.array([6, 9, -1, 6], dtype="Int32")
    raw["time"] = pd.array(["08:30", "23:10", "bad", "00:05"], dtype="string")
    out, report = clean_collisions(raw, codebook)
    assert len(out) == 4
    assert out.columns[:10].tolist() == [
        "collision_index",
        "collision_year",
        "date",
        "hour",
        "month",
        "day_of_year",
        "iso_week",
        "target",
        "severity3",
        "police_attended",
    ]
    assert out["target"].tolist() == [1, 1, 0, 0]
    assert out["police_attended"].isna().tolist() == [False, False, False, True]
    assert out["road_type"].isna().tolist() == [False, True, True, False]
    assert report.n_invalid_time == 1
    for leaked in ("number_of_casualties", "collision_severity", "enhanced_severity_collision"):
        assert leaked not in out.columns


def test_clean_collisions_rejects_duplicate_index(codebook: Codebook) -> None:
    raw = make_raw_collisions(3)
    raw.loc[2, "collision_index"] = raw.loc[0, "collision_index"]
    with pytest.raises(ValueError, match="duplicated"):
        clean_collisions(raw, codebook)
