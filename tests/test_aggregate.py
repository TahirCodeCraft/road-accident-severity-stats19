"""Tests for collision-level vehicle and casualty aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.aggregate import (
    VehicleAggConfig,
    aggregate_vehicles,
    attach_aggregates,
    count_codes,
    numeric_summaries,
    pedestrian_features,
)
from src.data.schema import PED_FEATURES
from tests.conftest import make_vehicles


@pytest.fixture
def vehicles() -> pd.DataFrame:
    return make_vehicles(seed=1)


def test_one_row_per_collision(vehicles: pd.DataFrame) -> None:
    agg = aggregate_vehicles(vehicles, VehicleAggConfig(fit_year=2023))
    assert agg.index.is_unique
    assert set(agg.index) == set(vehicles["collision_index"])
    sizes = vehicles.groupby("collision_index").size()
    assert (agg["n_vehicle_rows"] == sizes.reindex(agg.index)).all()


def test_permutation_invariance(vehicles: pd.DataFrame) -> None:
    base = aggregate_vehicles(vehicles)
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(len(vehicles))
        shuffled = vehicles.iloc[perm].reset_index(drop=True)
        # Re-label vehicles *within* each collision (keys stay unique, order/labels change).
        shuffled["vehicle_reference"] = (
            shuffled.groupby("collision_index")["vehicle_reference"]
            .transform(lambda s, seed=seed: s.sample(frac=1, random_state=seed).to_numpy())
        )
        pd.testing.assert_frame_equal(base, aggregate_vehicles(shuffled))


def test_counts_use_fit_levels_other_and_missing() -> None:
    frame = pd.DataFrame(
        {
            "collision_index": pd.array(["A", "A", "A", "B", "B"], dtype="string"),
            "collision_year": pd.array([2023, 2023, 2023, 2024, 2024], dtype="Int32"),
            "vehicle_type": pd.array([9, 9, None, 9, 42], dtype="Int32"),
        }
    )
    out = count_codes(frame, "vehicle_type", [9], "veh_")
    assert list(out.columns) == [
        "veh_vehicle_type__9",
        "veh_vehicle_type__other",
        "veh_vehicle_type__missing",
    ]
    assert out.loc["A"].tolist() == [2, 0, 1]
    assert out.loc["B"].tolist() == [1, 1, 0]


def test_unseen_code_goes_to_other(vehicles: pd.DataFrame) -> None:
    mutated = vehicles.copy()
    mask = mutated["collision_year"].eq(2024)
    mutated.loc[mask, "vehicle_type"] = 77  # never seen in 2023
    agg = aggregate_vehicles(mutated)
    assert "veh_vehicle_type__77" not in agg.columns
    n_2024 = int(mask.sum())
    assert int(agg["veh_vehicle_type__other"].sum()) == n_2024


def test_numeric_summaries() -> None:
    frame = pd.DataFrame(
        {
            "collision_index": pd.array(["A", "A", "B"], dtype="string"),
            "age_of_driver": pd.array([20, 40, None], dtype="Int32"),
        }
    )
    out = numeric_summaries(frame, ["age_of_driver"], "veh_")
    assert out.loc["A"].tolist() == [20.0, 30.0, 40.0]
    assert out.loc["B"].isna().all()


def test_tier_b_prefixed(vehicles: pd.DataFrame) -> None:
    agg = aggregate_vehicles(vehicles)
    assert any(c.startswith("tierb_first_point_of_impact__") for c in agg.columns)
    assert not any(c.startswith("veh_first_point_of_impact") for c in agg.columns)


def _casualties() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collision_index": pd.array(["A", "A", "A", "B", "C"], dtype="string"),
            "casualty_reference": pd.array([3, 1, 2, 1, 1], dtype="Int32"),
            "casualty_class": pd.array([3, 1, 3, 1, 3], dtype="Int32"),
            "age_of_casualty": pd.array([70, 30, 8, 25, None], dtype="Int32"),
            "pedestrian_location": pd.array([5, 0, 1, 0, None], dtype="Int32"),
            "pedestrian_movement": pd.array([3, 0, 1, 0, 2], dtype="Int32"),
        }
    )


def test_pedestrian_features_first_pedestrian() -> None:
    cas = _casualties()
    out = pedestrian_features(cas)
    assert list(out.columns) == list(PED_FEATURES)
    assert out.loc["A", "ped_involved"] == 1 and out.loc["B", "ped_involved"] == 0
    assert out.loc["A", "ped_age"] == 8  # casualty_reference 2 < 3
    assert out.loc["A", "ped_location"] == 1
    assert pd.isna(out.loc["B", "ped_age"])
    assert pd.isna(out.loc["C", "ped_age"]) and out.loc["C", "ped_movement"] == 2
    shuffled = cas.iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(out, pedestrian_features(shuffled))


def test_no_casualty_counts_and_row_count_preserved(vehicles: pd.DataFrame) -> None:
    ids = sorted(set(vehicles["collision_index"])) + ["NOVEH"]
    collisions = pd.DataFrame({"collision_index": pd.array(ids, dtype="string")})
    cas = _casualties()
    cas["collision_index"] = pd.array(ids[:3] + ids[:1] + ids[2:3], dtype="string")
    out = attach_aggregates(collisions, aggregate_vehicles(vehicles), pedestrian_features(cas))
    assert len(out) == len(collisions)
    assert out["collision_index"].is_unique
    ped_cols = [c for c in out.columns if c.startswith("ped_") or "casualt" in c]
    assert sorted(ped_cols) == sorted(PED_FEATURES)
    row = out.set_index("collision_index").loc["NOVEH"]
    assert row["n_vehicle_rows"] == 0 and row["ped_involved"] == 0


def test_duplicate_vehicle_keys_rejected(vehicles: pd.DataFrame) -> None:
    import pytest

    dup = pd.concat([vehicles, vehicles.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        aggregate_vehicles(dup)
