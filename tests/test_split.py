"""Tests for the temporal split and the test-set guard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.clean import date_parts, parse_dates
from src.data.split import (
    SPLITS,
    TestSplitLockedError,
    assign_split,
    load_split,
    split_summary,
)


def _frame() -> pd.DataFrame:
    dates = pd.Series(
        [
            "01/01/2023",
            "15/06/2023",
            "31/12/2023",
            "01/01/2024",  # ISO week 1 -> tune
            "08/01/2024",  # ISO week 2 -> cal
            "30/12/2024",  # ISO week 1 (of 2025) -> tune
            "23/12/2024",  # ISO week 52 -> cal
            "01/01/2025",
            "31/12/2025",
        ]
    )
    parsed = parse_dates(dates)
    frame = pd.concat([pd.DataFrame({"date": parsed}), date_parts(parsed)], axis=1)
    frame["collision_year"] = parsed.dt.year.astype("Int32")
    frame["collision_index"] = [f"C{i}" for i in range(len(frame))]
    frame["target"] = [1, 0, 0, 1, 0, 1, 0, 1, 0]
    return frame


def test_assign_split_rules() -> None:
    frame = _frame()
    split = assign_split(frame)
    assert split.tolist() == ["fit", "fit", "fit", "tune", "cal", "tune", "cal", "test", "test"]


def test_splits_disjoint_and_test_is_2025_only() -> None:
    frame = _frame()
    frame["split"] = assign_split(frame)
    groups = {s: set(frame.loc[frame["split"] == s, "collision_index"]) for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            assert not groups[a] & groups[b]
    assert set().union(*groups.values()) == set(frame["collision_index"])
    assert set(frame.loc[frame["split"] == "test", "collision_year"]) == {2025}
    assert 2025 not in set(frame.loc[frame["split"] != "test", "collision_year"])
    assert set(frame.loc[frame["split"].isin(["tune", "cal"]), "collision_year"]) == {2024}


def test_assign_split_rejects_other_years() -> None:
    frame = _frame()
    frame.loc[0, "collision_year"] = 2022
    with pytest.raises(ValueError):
        assign_split(frame)


@pytest.fixture
def parquet(tmp_path: Path) -> Path:
    path = tmp_path / "collisions.parquet"
    _frame().to_parquet(path, index=False)
    return path


def test_load_split_dev(parquet: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "frozen_manifest.json"
    fit = load_split("fit", path=parquet, manifest_path=manifest)
    assert len(fit) == 3 and set(fit["split"]) == {"fit"}
    fit_tune = load_split("fit+tune", path=parquet, manifest_path=manifest)
    assert len(fit_tune) == 5
    with pytest.raises(ValueError):
        load_split("train", path=parquet, manifest_path=manifest)


def test_test_guard(parquet: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "frozen_manifest.json"
    with pytest.raises(TestSplitLockedError):
        load_split("test", path=parquet, manifest_path=manifest)
    with pytest.raises(TestSplitLockedError):
        load_split("test", allow_test=True, path=parquet, manifest_path=manifest)
    manifest.write_text("{}")
    with pytest.raises(TestSplitLockedError):
        load_split("test", allow_test=False, path=parquet, manifest_path=manifest)
    with pytest.raises(TestSplitLockedError):
        load_split("cal+test", path=parquet, manifest_path=manifest)
    test = load_split("test", allow_test=True, path=parquet, manifest_path=manifest)
    assert set(test["collision_year"]) == {2025}


def test_split_summary() -> None:
    summary = split_summary(_frame())
    assert summary["split"].tolist() == list(SPLITS)
    assert summary["rows"].tolist() == [3, 2, 2, 2]
    assert summary.set_index("split").loc["fit", "severe_rate"] == pytest.approx(1 / 3, abs=1e-4)
