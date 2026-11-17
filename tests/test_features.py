"""Tests for feature engineering, encoders, feature-frame contract and selection."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_phase_frames
from src.features.encoders import FrequencyEncoder, OOFTargetEncoder
from src.features.engineered import (
    BankHolidays,
    EngineeredConfig,
    collision_level_features,
    environment_flags,
    pedestrian_flags,
    road_flags,
    temporal_features,
    vehicle_mix_features,
)
from src.features.frame import (
    FEATURES_DIR,
    ID_COLS,
    assemble,
    feature_columns,
    forbidden_columns,
    load_collisions,
)
from src.features.selection import (
    SelectionConfig,
    resolve_redundancy,
    select_features,
    top_value_shares,
)
from src.paths import COLLISIONS_PATH, VEHICLES_CLEAN_PATH

REAL_DATA = COLLISIONS_PATH.exists() and VEHICLES_CLEAN_PATH.exists()
needs_data = pytest.mark.skipif(not REAL_DATA, reason="processed data not built")
FORBIDDEN_EXACT = {"police_attended", "iso_week", "collision_year", "date", "severity3",
                   "local_authority_ons_district", "number_of_casualties"}
CONFIG = EngineeredConfig()
HOLIDAYS = BankHolidays(
    england_and_wales=frozenset({dt.date(2023, 5, 8)}),
    scotland=frozenset({dt.date(2023, 8, 7)}),
    scotland_police_forces=frozenset({99}),
)


# --- fixtures ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small stratified sample of the real processed tables (all four splits)."""
    if not REAL_DATA:
        pytest.skip("processed data not built")
    collisions = load_collisions()
    parts = [g.sample(n=600, random_state=0) for _, g in collisions.groupby("split")]
    rows = pd.concat(parts).reset_index(drop=True)
    vehicles = pd.read_parquet(VEHICLES_CLEAN_PATH)
    vehicles = vehicles[vehicles["collision_index"].isin(set(rows["collision_index"]))]
    return rows, vehicles.reset_index(drop=True)


@pytest.fixture(scope="module")
def sample_frames(sample: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
    """Assembled F0 / F1full frames of both phases built from the sample."""
    rows, vehicles = sample
    mix = vehicle_mix_features(vehicles)
    return {phase: build_phase_frames(rows, mix, phase, CONFIG) for phase in ("tune", "refit")}


def make_rows(n: int, **overrides: object) -> pd.DataFrame:
    """Hand-made collision rows with neutral defaults (daylight, dry, 30 mph urban, ...)."""
    base: dict[str, object] = {
        "collision_index": [f"C{i}" for i in range(n)],
        "date": pd.Timestamp("2023-03-15"), "hour": 12, "month": 3, "day_of_week": 4,
        "police_force": 1, "light_conditions": 1, "weather_conditions": 1,
        "road_surface_conditions": 1, "speed_limit": 30, "urban_or_rural_area": 1,
        "road_type": 6, "junction_detail": 0, "junction_control": 0, "first_road_class": 3,
        "second_road_class": 0, "pedestrian_crossing": 0, "carriageway_hazards": 0,
        "ped_involved": 0, "ped_age": pd.NA, "ped_location": pd.NA, "number_of_vehicles": 2,
        "longitude": -1.0, "latitude": 52.0,
    }
    base.update(overrides)
    frame = pd.DataFrame({k: (v if isinstance(v, list) else [v] * n) for k, v in base.items()})
    for col in frame.columns:
        if col not in ("collision_index", "date", "longitude", "latitude"):
            frame[col] = frame[col].astype("Int32")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def make_vehicles(records: list[tuple]) -> pd.DataFrame:
    """Vehicle rows from ``(collision, ref, type, manoeuvre, from, to, age, sex)`` tuples."""
    cols = ["collision_index", "vehicle_reference", "vehicle_type", "vehicle_manoeuvre",
            "vehicle_direction_from", "vehicle_direction_to", "age_of_driver", "sex_of_driver"]
    frame = pd.DataFrame(records, columns=cols)
    for col in cols[1:]:
        frame[col] = frame[col].astype("Int32")
    frame["escooter_flag"] = pd.array([0] * len(frame), dtype="Int32")
    frame["journey_purpose_of_driver"] = pd.array([pd.NA] * len(frame), dtype="Int32")
    frame["propulsion_code"] = pd.array([1] * len(frame), dtype="Int32")
    frame["driver_imd_decile"] = pd.array([5] * len(frame), dtype="Int32")
    return frame


# --- encoders ---------------------------------------------------------------------------------
def test_frequency_encoder_unseen_and_missing_are_zero() -> None:
    enc = FrequencyEncoder().fit(pd.Series(["a", "a", "b", None]))
    out = enc.transform(pd.Series(["a", "b", "zzz", None]))
    np.testing.assert_allclose(out.to_numpy(), [0.5, 0.25, 0.0, 0.0])
    assert out.dtype == np.float32


def test_target_encoder_m_estimate_and_unseen_prior() -> None:
    keys = pd.Series(["a"] * 10 + ["b"] * 10)
    y = np.array([1] * 10 + [0] * 10)
    enc = OOFTargetEncoder(m=10).fit(keys, y)
    expected_a = (10 + 10 * 0.5) / (10 + 10)
    out = enc.transform(pd.Series(["a", "new", None]))
    np.testing.assert_allclose(out.to_numpy(), [expected_a, 0.5, 0.5], rtol=1e-6)


def test_oof_target_encoding_differs_from_in_sample() -> None:
    rng = np.random.default_rng(0)
    keys = pd.Series(rng.choice(list("abcdefgh"), size=2000))
    y = rng.integers(0, 2, size=2000)
    enc = OOFTargetEncoder(m=50, n_splits=5, seed=1)
    oof = enc.fit_transform(keys, y)
    in_sample = enc.transform(keys)
    assert not np.allclose(oof.to_numpy(), in_sample.to_numpy())
    # OOF values never use the row's own label: flipping one label leaves its value unchanged.
    y_flip = y.copy()
    y_flip[0] = 1 - y_flip[0]
    oof_flip = OOFTargetEncoder(m=50, n_splits=5, seed=1).fit_transform(keys, y_flip)
    assert oof_flip.iloc[0] == pytest.approx(oof.iloc[0])


# --- frame contract ---------------------------------------------------------------------------
@needs_data
def test_encoders_fit_only_on_training_rows(sample: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    rows, vehicles = sample
    mix = vehicle_mix_features(vehicles)
    for phase, flip_splits in (("tune", ("tune", "cal")), ("refit", ("cal", "test"))):
        base = build_phase_frames(rows, mix, phase, CONFIG)
        flipped = rows.copy()
        mask = flipped["split"].isin(flip_splits)
        flipped.loc[mask, "target"] = (1 - flipped.loc[mask, "target"]).astype(np.int8)
        other = build_phase_frames(flipped, mix, phase, CONFIG)
        for name in ("F0", "F1full"):
            feats = feature_columns(base[name])
            pd.testing.assert_frame_equal(base[name][feats], other[name][feats])
    # Sanity: flipping FIT labels does change the target encodings.
    flipped = rows.copy()
    fit = flipped["split"].eq("fit")
    flipped.loc[fit, "target"] = (1 - flipped.loc[fit, "target"]).astype(np.int8)
    changed = build_phase_frames(flipped, mix, "tune", CONFIG)["F1full"]["te_police_force"]
    assert not np.allclose(changed, build_phase_frames(rows, mix, "tune", CONFIG)["F1full"]
                           ["te_police_force"])


@needs_data
def test_tune_phase_has_no_test_rows(sample_frames: dict) -> None:
    for frame in sample_frames["tune"].values():
        assert set(frame["split"]) == {"fit", "tune", "cal"}
    for frame in sample_frames["refit"].values():
        assert set(frame["split"]) == {"fit", "tune", "cal", "test"}


@needs_data
def test_no_deny_listed_columns_in_sample_frames(sample_frames: dict) -> None:
    for frames in sample_frames.values():
        for frame in frames.values():
            assert list(frame.columns[:3]) == list(ID_COLS)
            feats = feature_columns(frame)
            assert not forbidden_columns(feats)
            assert not FORBIDDEN_EXACT & set(feats)
            assert not [c for c in feats if c.startswith("tierb_")]


def test_assemble_rejects_deny_listed_column() -> None:
    rows = pd.DataFrame({"collision_index": ["a"], "split": ["fit"], "target": [1]})
    feats = pd.DataFrame({"tierb_first_point_of_impact__1": np.float32([1.0])})
    with pytest.raises(ValueError, match="Deny-listed"):
        assemble(rows, feats)


@needs_data
def test_categorical_dtypes_survive_parquet_round_trip(sample_frames: dict,
                                                       tmp_path: Path) -> None:
    frame = sample_frames["refit"]["F1full"]
    path = tmp_path / "frame.parquet"
    frame.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    for col in feature_columns(frame):
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            assert isinstance(back[col].dtype, pd.CategoricalDtype), col
            assert list(back[col].cat.categories) == list(frame[col].cat.categories), col
        else:
            assert back[col].dtype == np.float32, col
    pd.testing.assert_frame_equal(frame[feature_columns(frame)], back[feature_columns(frame)],
                                  check_categorical=True)


@pytest.mark.skipif(not FEATURES_DIR.exists(), reason="feature parquets not built")
def test_written_feature_parquets_respect_contract() -> None:
    import pyarrow.parquet as pq

    paths = sorted(FEATURES_DIR.glob("*_*.parquet"))
    assert paths
    for path in paths:
        names = pq.read_schema(path).names
        feats = [c for c in names if c not in ID_COLS]
        assert names[:3] == list(ID_COLS), path.name
        if path.stem.startswith("F1S1_"):
            # Sensitivity S1 deliberately adds the Tier-B crash-mechanics fields; nothing else.
            feats = [c for c in feats if not c.startswith("tierb_")]
        assert not forbidden_columns(feats), path.name
        assert not FORBIDDEN_EXACT & set(feats), path.name
        splits = set(pq.read_table(path, columns=["split"]).column("split").to_pylist())
        if path.stem.endswith("_tune"):
            assert "test" not in splits, path.name
        sample = pd.read_parquet(path).head(5)
        for col in feats:
            dtype = sample[col].dtype
            assert isinstance(dtype, pd.CategoricalDtype) or dtype == np.float32, (path, col)


# --- vehicle aggregation ----------------------------------------------------------------------
@needs_data
def test_vehicle_aggregates_order_invariant(sample: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, vehicles = sample
    base = vehicle_mix_features(vehicles).sort_index()
    for seed in range(3):
        shuffled = vehicles.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        shuffled["vehicle_reference"] = shuffled["vehicle_reference"].sample(
            frac=1.0, random_state=seed + 10).to_numpy()
        other = vehicle_mix_features(shuffled).sort_index()
        pd.testing.assert_frame_equal(base, other, check_exact=False, rtol=1e-9)


def test_vehicle_mix_hand_made() -> None:
    vehicles = make_vehicles([
        # C0: car northbound (S->N) vs HGV southbound (N->S): head-on, mass 3 vs 6
        ("C0", 1, 9, 19, 5, 1, 30, 1),
        ("C0", 2, 21, 13, 1, 5, 55, 2),
        # C1: pedal cycle eastbound (W->E) vs car northbound: crossing (90 deg)
        ("C1", 1, 1, 9, 7, 3, 16, 1),
        ("C1", 2, 9, 2, 5, 1, pd.NA, pd.NA),
        # C2: single motorcycle, unknown manoeuvre
        ("C2", 1, 5, pd.NA, 1, 5, 72, 1),
    ])
    mix = vehicle_mix_features(vehicles)
    assert mix.loc["C0", "relative_heading"] == pytest.approx(180.0)
    assert mix.loc["C1", "relative_heading"] == pytest.approx(90.0)
    assert np.isnan(mix.loc["C2", "relative_heading"])
    assert (mix.loc["C0", "max_mass_class"], mix.loc["C0", "min_mass_class"]) == (6, 3)
    assert mix.loc["C0", "mass_mismatch"] == 3
    assert mix.loc["C0", ["n_car", "n_hgv"]].tolist() == [1, 1]
    assert mix.loc["C1", "n_pedal_cycle"] == 1 and mix.loc["C1", "has_vru_vehicle"] == 1
    assert mix.loc["C0", "has_vru_vehicle"] == 0
    assert mix.loc["C2", "n_motorcycle"] == 1
    assert mix.loc["C0", "any_overtaking"] == 1 and mix.loc["C1", "any_overtaking"] == 0
    assert mix.loc["C1", "any_right_turn"] == 1 and mix.loc["C1", "any_stationary"] == 1
    assert np.isnan(mix.loc["C2", "any_overtaking"])  # manoeuvre unknown for every vehicle
    assert mix.loc["C1", "any_young_driver"] == 1 and mix.loc["C0", "any_young_driver"] == 0
    assert mix.loc["C2", "any_older_driver"] == 1
    assert mix.loc["C1", "n_age_unknown"] == 1
    assert mix.loc["C0", "driver_age_range"] == 25
    assert mix.loc["C0", "share_male_driver"] == pytest.approx(0.5)
    assert mix.loc["C1", "share_male_driver"] == pytest.approx(1.0)
    assert np.isnan(mix.loc["C0", "any_work_journey"])


def test_escooter_flag_overrides_vehicle_type() -> None:
    vehicles = make_vehicles([("C0", 1, 90, 19, 1, 5, 20, 1), ("C0", 2, 9, 19, 5, 1, 40, 1)])
    vehicles.loc[0, "escooter_flag"] = 1
    mix = vehicle_mix_features(vehicles)
    assert mix.loc["C0", "n_escooter"] == 1 and mix.loc["C0", "n_other"] == 0
    assert mix.loc["C0", "has_vru_vehicle"] == 1 and mix.loc["C0", "min_mass_class"] == 1


# --- engineered flags on hand-made rows -------------------------------------------------------
def test_temporal_flags() -> None:
    rows = make_rows(
        5,
        # Fri 23h, Sat 03h, Wed 08h, Mon 8 May 2023 (E&W holiday), Scottish 7 Aug 2023
        day_of_week=[6, 7, 4, 2, 2], hour=[23, 3, 8, 12, 12], month=[1, 1, 3, 5, 8],
        date=[pd.Timestamp("2023-01-06"), pd.Timestamp("2023-01-07"),
              pd.Timestamp("2023-03-15"), pd.Timestamp("2023-05-08"),
              pd.Timestamp("2023-08-07")],
        police_force=[1, 1, 1, 1, 99],
    )
    out = temporal_features(rows, HOLIDAYS)
    assert out["is_night"].tolist() == [1, 1, 0, 0, 0]
    assert out["weekend_night"].tolist() == [1, 1, 0, 0, 0]
    assert out["is_weekend"].tolist() == [0, 1, 0, 0, 0]
    assert out["rush_hour"].tolist() == [0, 0, 1, 0, 0]
    assert out["is_bank_holiday"].tolist() == [0, 0, 0, 1, 1]
    assert out["season"].astype(str).tolist() == ["winter", "winter", "spring", "spring",
                                                  "summer"]
    assert out["hour_sin"].iloc[2] == pytest.approx(np.sin(2 * np.pi * 8 / 24), rel=1e-5)
    # Scottish force on an England-only holiday is not a holiday.
    rows2 = make_rows(1, date=pd.Timestamp("2023-05-08"), police_force=99)
    assert temporal_features(rows2, HOLIDAYS)["is_bank_holiday"].iloc[0] == 0


def test_environment_flags() -> None:
    rows = make_rows(
        4,
        light_conditions=[6, 4, 1, pd.NA], weather_conditions=[7, 5, 1, 2],
        road_surface_conditions=[2, 1, 2, 1],
    )
    env = {k: v.astype("Float64").tolist() for k, v in environment_flags(rows).items()}
    assert env["is_dark"][:3] == [1, 1, 0] and pd.isna(env["is_dark"][3])
    assert env["dark_unlit"][:3] == [1, 0, 0]
    assert env["high_wind"] == [0, 1, 0, 0]
    assert env["poor_weather"] == [1, 1, 0, 1]
    assert env["wet_or_icy"] == [1, 0, 1, 0]
    assert env["dark_and_wet"][:3] == [1, 0, 0]
    assert env["fine_but_wet"] == [0, 0, 1, 0]
    assert env["visibility_risk"][:3] == [1, 1, 0]


def test_road_flags() -> None:
    rows = make_rows(
        3,
        speed_limit=[60, 70, 30], urban_or_rural_area=[2, 1, 1], road_type=[6, 3, 1],
        junction_detail=[0, 16, 13], junction_control=[0, 2, 4], first_road_class=[3, 1, 6],
        pedestrian_crossing=[0, 0, 13], carriageway_hazards=[0, 13, 0],
    )
    road = {k: pd.Series(v).astype("Float64").tolist() for k, v in road_flags(rows).items()}
    assert road["high_speed"] == [1, 1, 0]
    assert road["rural_high_speed"] == [1, 0, 0]
    assert road["rural_single_carriageway_60"] == [1, 0, 0]
    assert road["is_junction"] == [0, 1, 1]
    assert road["junction_complexity"] == [0, 3, 2]
    assert road["uncontrolled_junction"] == [0, 0, 1]
    assert road["is_roundabout"] == [0, 0, 1]
    assert road["is_motorway"] == [0, 1, 0]
    assert road["crossing_present"] == [0, 0, 1]
    assert road["hazard_present"] == [0, 1, 0]


def test_pedestrian_flags() -> None:
    rows = make_rows(3, ped_involved=[1, 1, 0], ped_age=[8, 75, pd.NA],
                     ped_location=[5, 1, pd.NA])
    flags, group = pedestrian_flags(rows)
    assert flags["ped_child"].astype("Float64").tolist() == [1, 0, 0]
    assert flags["ped_elderly"].astype("Float64").tolist() == [0, 1, 0]
    assert list(group.astype(str)) == ["crossing_elsewhere", "on_crossing", "no_pedestrian"]


def test_interactions_and_categoricals() -> None:
    rows = make_rows(
        2, speed_limit=[60, 30], number_of_vehicles=[1, 2], urban_or_rural_area=[2, 1],
        light_conditions=[4, 1], ped_involved=[1, 0], ped_age=[40, pd.NA],
        ped_location=[5, pd.NA], first_road_class=[3, 6], second_road_class=[0, 4],
    )
    vehicles = make_vehicles([("C0", 1, 9, 19, 1, 5, 30, 1),
                              ("C1", 1, 9, 19, 1, 5, 30, 1), ("C1", 2, 11, 19, 5, 1, 50, 2)])
    block = collision_level_features(rows, vehicle_mix_features(vehicles), HOLIDAYS, CONFIG)
    assert block["has_vru"].tolist() == [1, 0]  # pedestrian counts as VRU
    assert block["high_speed_x_vru"].tolist() == [1, 0]
    assert block["dark_x_ped"].tolist() == [1, 0]
    assert block["rural_x_single_vehicle"].tolist() == [1, 0]
    assert block["speed_x_single_vehicle"].tolist() == [60, 0]
    assert block["road_class_pair"].astype(str).tolist() == ["A-none", "U-B"]
    assert block["heading_class"].astype(str).tolist()[1] == "head_on"
    assert block["single_vehicle"].tolist() == [1, 0]
    assert all(block[c].dtype == np.float32 or isinstance(block[c].dtype, pd.CategoricalDtype)
               for c in block.columns)


# --- selection --------------------------------------------------------------------------------
def test_resolve_redundancy_keeps_higher_mi() -> None:
    mi = pd.Series({"a": 0.1, "b": 0.2, "c": 0.05})
    dropped = resolve_redundancy([("a", "b", 0.99), ("b", "c", 0.97)], mi)
    assert set(dropped) == {"a", "c"}
    assert dropped["a"][0] == "b"


def test_top_value_share_counts_nan() -> None:
    X = pd.DataFrame({"x": [np.nan] * 999 + [1.0]})
    assert top_value_shares(X)["x"] == pytest.approx(0.999)


def test_select_features_end_to_end_small() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.7, size=n) > 0.8).astype(int)
    X = pd.DataFrame({
        "month": rng.integers(1, 13, size=n).astype(np.float32),
        "signal": signal.astype(np.float32),
        "signal_copy": (signal * 2 + 1).astype(np.float32),
        "constant": np.zeros(n, dtype=np.float32),
        "noise": rng.normal(size=n).astype(np.float32),
        "cat": pd.Categorical(rng.choice(["a", "b"], size=n)),
    })
    config = SelectionConfig(n_repeats=2, n_jobs=2,
                             lgbm_params={"n_estimators": 40, "learning_rate": 0.1,
                                          "num_leaves": 15, "verbose": -1})
    result = select_features(X, y, config)
    table = result.table
    assert table.loc["constant", "stage"] == "near_constant"
    assert {"signal", "signal_copy"} & set(table.index[table["stage"].eq("redundancy")])
    assert ("signal" in result.selected) or ("signal_copy" in result.selected)
    assert all(table.loc[f, "reason"] for f in table.index if f not in result.selected)
