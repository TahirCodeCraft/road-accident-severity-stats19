"""F1-full feature set: F0 plus the engineered features of plan section 3.

Groups (code meanings from ``data/raw/data-guide-2025.xlsx``):

* temporal: hour / month sin-cos, weekend, night (22-05), rush hour, weekend night, season,
  bank holiday (static list in ``configs/features/bank_holidays.yaml``).
* environment: darkness, unlit darkness, poor weather, high wind, wet/icy surface and their
  combinations.
* road: high speed, rural high speed, rural 60 mph single carriageway, junction flags and
  complexity, roundabout, motorway, road-class pair, crossing and hazard presence.
* vehicle mix (order-free aggregation of ``vehicles_clean``): counts per vehicle group, VRU
  involvement, mass classes, relative heading of two-vehicle collisions, manoeuvre flags.
* driver: age range, young / older drivers, unknown ages, male share, work journey, electric
  propulsion, home-area deprivation.
* pedestrian: child / elderly pedestrian, pedestrian location group.
* interactions and geography (out-of-fold target encodings, rotated coordinates).

Missing semantics: a flag is NaN when its source field is missing (Kleene logic for
combinations). Vehicle "any_*" flags are 1 if any vehicle with a known value has the property,
0 if none does, and NaN only when the field is unknown for every vehicle of the collision.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.encoders import OOFTargetEncoder
from src.features.frame import KEY_COL, to_category, to_float32
from src.paths import CONFIGS_DIR

logger = logging.getLogger(__name__)

BANK_HOLIDAYS_PATH: Path = CONFIGS_DIR / "features" / "bank_holidays.yaml"

# --- code sets (data guide) -------------------------------------------------------------------
SUNDAY, SATURDAY = 1, 7
DARK_LIGHT = (4, 5, 6, 7)  # darkness: lit / unlit / no lighting / lighting unknown
DARK_UNLIT_LIGHT = (5, 6)  # darkness: lights unlit / no lighting
POOR_WEATHER = (2, 3, 4, 5, 6, 7)  # rain, snow, high winds, fog (1 fine, 8 other)
HIGH_WIND_WEATHER = (4, 5, 6)
RAIN_SNOW_WEATHER = (2, 3, 5, 6)
FOG_WEATHER = (7,)
FINE_WEATHER = (1,)
WET_ICY_SURFACE = (2, 3, 4, 5)  # wet/damp, snow, frost/ice, flood
RURAL = (2,)
SINGLE_CARRIAGEWAY = (6,)
ROUNDABOUT = (1,)
MOTORWAY_CLASS = (1, 2)  # Motorway, A(M)
NOT_AT_JUNCTION = 0
GIVE_WAY_UNCONTROLLED = (4,)
NO_CROSSING = 0
NO_HAZARD = 0
JUNCTION_COMPLEXITY = {0: 0, 18: 1, 13: 2, 19: 2, 16: 3, 17: 4}
ROAD_CLASS_NAMES = {0: "none", 1: "M", 2: "AM", 3: "A", 4: "B", 5: "C", 6: "U"}
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
SEASON_LEVELS = ("winter", "spring", "summer", "autumn")
HIGH_SPEED_MPH = 60

VEHICLE_GROUPS: dict[str, tuple[int, ...]] = {
    "car": (8, 9),  # taxi / private hire, car
    "van": (19, 98),  # goods <=3.5t; goods unknown weight (median engine ~2.2 l, van-like)
    "hgv": (20, 21),  # goods 3.5-7.5t, goods >=7.5t
    "bus": (10, 11),  # minibus, bus or coach
    "motorcycle": (2, 3, 4, 5, 23, 97),  # all motorcycles incl. electric / unknown cc
    "pedal_cycle": (1,),
    "escooter": (33,),  # personal powered transporter (+ escooter_flag == 1)
}
OTHER_GROUP = "other"  # ridden horse, agricultural, tram, mobility scooter, other, unknown
ESCOOTER_GROUP = "escooter"
VRU_TYPES = (1, 2, 3, 4, 5, 23, 97, 16, 22, 33)  # cycles, motorcycles, horse, mobility, PPT
# Ordinal mass class: 1 cycle/scooter/horse, 2 motorcycle, 3 car, 4 van/minibus,
# 5 medium goods/agricultural, 6 HGV >=7.5t / bus / tram. "Other vehicle" and unknown -> NaN.
MASS_CLASS: dict[int, int] = {
    1: 1, 16: 1, 22: 1, 33: 1,
    2: 2, 3: 2, 4: 2, 5: 2, 23: 2, 97: 2,
    8: 3, 9: 3,
    10: 4, 19: 4, 98: 4,
    17: 5, 20: 5,
    11: 6, 18: 6, 21: 6,
}
HEAVY_MASS_CLASS = 5
OVERTAKING = (13, 14, 15)
RIGHT_TURN = (9, 10)  # turning right, waiting to turn right
STATIONARY = (2, 3, 8, 10)  # parked, waiting to go ahead / turn left / turn right
REVERSING = (1,)
WORK_JOURNEY = (1,)  # journey as part of work
ELECTRIC = (3,)
MALE, FEMALE = 1, 2
YOUNG_AGE, OLDER_AGE = 25, 70
CHILD_AGE = 16
PED_CROSSING_GROUPS: dict[int, str] = {
    1: "on_crossing", 2: "on_crossing", 3: "on_crossing", 4: "near_crossing",
    5: "crossing_elsewhere", 6: "footway_or_refuge", 7: "footway_or_refuge",
    8: "in_carriageway", 9: "in_carriageway",
}
NO_PEDESTRIAN = "no_pedestrian"
PED_GROUP_LEVELS = (NO_PEDESTRIAN, "on_crossing", "near_crossing", "crossing_elsewhere",
                    "footway_or_refuge", "in_carriageway")
HEADING_LEVELS = ("same_direction", "crossing", "head_on")
COMPASS_CODES = tuple(range(1, 9))  # 1 N, 2 NE, ... 8 NW (45 degree steps)
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_EQUATOR = 111.32


@dataclass(frozen=True)
class EngineeredConfig:
    """Configuration of the F1 engineered features.

    Attributes:
        te_m: m-estimate smoothing of target encodings.
        te_folds: Out-of-fold splits for training rows.
        seed: Seed of the out-of-fold assignment.
        grid_km: Grid-cell size (km) of the geographic target encoding.
        origin_lon: Projection origin longitude (degrees).
        origin_lat: Projection origin latitude (degrees).
        rotation_angles: Angles (degrees) of the rotated coordinate features.
        bank_holidays_path: YAML file with the static bank-holiday lists.
    """

    te_m: float = 50.0
    te_folds: int = 5
    seed: int = 42
    grid_km: float = 5.0
    origin_lon: float = -2.0
    origin_lat: float = 54.0
    rotation_angles: tuple[int, ...] = (15, 30, 45, 60)
    bank_holidays_path: Path = BANK_HOLIDAYS_PATH


@dataclass
class BankHolidays:
    """Static bank-holiday calendars.

    Attributes:
        england_and_wales: Dates applied to non-Scottish forces.
        scotland: Dates applied to Scottish forces.
        scotland_police_forces: Police-force codes in Scotland.
    """

    england_and_wales: frozenset[dt.date] = field(default_factory=frozenset)
    scotland: frozenset[dt.date] = field(default_factory=frozenset)
    scotland_police_forces: frozenset[int] = field(default_factory=frozenset)


def load_bank_holidays(path: Path = BANK_HOLIDAYS_PATH) -> BankHolidays:
    """Read the bank-holiday YAML.

    Args:
        path: YAML path.

    Returns:
        Parsed :class:`BankHolidays`.
    """
    raw = yaml.safe_load(path.read_text())
    return BankHolidays(
        england_and_wales=frozenset(pd.Timestamp(d).date() for d in raw["england_and_wales"]),
        scotland=frozenset(pd.Timestamp(d).date() for d in raw["scotland"]),
        scotland_police_forces=frozenset(int(c) for c in raw["scotland_police_forces"]),
    )


# --- small helpers ----------------------------------------------------------------------------
def code_in(values: pd.Series, codes: Iterable[int]) -> pd.Series:
    """Nullable-boolean membership test (NA where the source is NA).

    Args:
        values: Integer code series (nullable).
        codes: Codes counted as True.

    Returns:
        ``boolean`` series.
    """
    hit = values.isin(list(codes)).astype("boolean")
    return hit.mask(values.isna())


def numeric(values: pd.Series) -> pd.Series:
    """Return a nullable Float64 copy of a numeric series."""
    return values.astype("Float64")


def _f32_frame(columns: dict[str, pd.Series | np.ndarray], index: pd.Index) -> pd.DataFrame:
    """Build a float32 frame from nullable / boolean columns."""
    return pd.DataFrame({k: to_float32(pd.Series(v)) for k, v in columns.items()}, index=index)


# --- temporal ---------------------------------------------------------------------------------
def bank_holiday_flag(rows: pd.DataFrame, holidays: BankHolidays) -> pd.Series:
    """Flag collisions on a bank holiday of their nation (Scotland vs England & Wales).

    Args:
        rows: Frame with ``date`` and ``police_force``.
        holidays: Parsed calendars.

    Returns:
        ``boolean`` series (NA if the date is missing).
    """
    dates = pd.to_datetime(rows["date"]).dt.date
    scottish = rows["police_force"].isin(list(holidays.scotland_police_forces)).fillna(False)
    ew_hit = dates.isin(holidays.england_and_wales)
    sc_hit = dates.isin(holidays.scotland)
    flag = pd.Series(np.where(scottish, sc_hit, ew_hit), index=rows.index).astype("boolean")
    return flag.mask(rows["date"].isna())


def temporal_features(rows: pd.DataFrame, holidays: BankHolidays) -> pd.DataFrame:
    """Cyclical time encodings and calendar flags.

    Args:
        rows: Frame with ``hour``, ``month``, ``day_of_week`` (1 = Sunday), ``date``,
            ``police_force``.
        holidays: Bank-holiday calendars.

    Returns:
        Frame of float32 features plus the ``season`` categorical.
    """
    hour, month = numeric(rows["hour"]), numeric(rows["month"])
    dow = rows["day_of_week"]
    weekend = code_in(dow, (SATURDAY, SUNDAY))
    night = (hour >= 22) | (hour <= 5)
    rush = ~weekend & (hour.between(7, 9) | hour.between(16, 18))
    fri_late, sun_early = code_in(dow, (6,)) & (hour >= 22), code_in(dow, (SUNDAY,)) & (hour <= 5)
    weekend_night = night & (fri_late | code_in(dow, (SATURDAY,)) | sun_early)
    out = _f32_frame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "is_weekend": weekend,
            "is_night": night,
            "rush_hour": rush,
            "weekend_night": weekend_night,
            "is_bank_holiday": bank_holiday_flag(rows, holidays),
        },
        rows.index,
    )
    out["season"] = to_category(rows["month"].map(SEASONS), SEASON_LEVELS)
    return out


# --- environment ------------------------------------------------------------------------------
def environment_flags(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """Light / weather / surface flags as nullable booleans.

    Args:
        rows: Frame with ``light_conditions``, ``weather_conditions``,
            ``road_surface_conditions``.

    Returns:
        ``{name: boolean series}``.
    """
    light, weather = rows["light_conditions"], rows["weather_conditions"]
    is_dark = code_in(light, DARK_LIGHT)
    dark_unlit = code_in(light, DARK_UNLIT_LIGHT)
    wet = code_in(rows["road_surface_conditions"], WET_ICY_SURFACE)
    visibility = code_in(weather, FOG_WEATHER) | dark_unlit
    visibility = visibility | (is_dark & code_in(weather, RAIN_SNOW_WEATHER))
    return {
        "is_dark": is_dark,
        "dark_unlit": dark_unlit,
        "poor_weather": code_in(weather, POOR_WEATHER),
        "high_wind": code_in(weather, HIGH_WIND_WEATHER),
        "wet_or_icy": wet,
        "dark_and_wet": is_dark & wet,
        "fine_but_wet": code_in(weather, FINE_WEATHER) & wet,
        "visibility_risk": visibility,
    }


# --- road -------------------------------------------------------------------------------------
def road_class_pair(rows: pd.DataFrame) -> pd.Categorical:
    """Categorical ``<first road class>-<second road class>`` (second ``none`` off-junction).

    Args:
        rows: Frame with ``first_road_class`` and ``second_road_class``.

    Returns:
        Categorical with a fixed vocabulary of all class pairs.
    """
    first = rows["first_road_class"].map(ROAD_CLASS_NAMES)
    second = rows["second_road_class"].map(ROAD_CLASS_NAMES)
    pair = (first.astype(object) + "-" + second.astype(object)).where(
        first.notna() & second.notna()
    )
    levels = [f"{a}-{b}" for a in ROAD_CLASS_NAMES.values() if a != "none"
              for b in ROAD_CLASS_NAMES.values()]
    return to_category(pair, levels)


def road_flags(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """Speed, carriageway and junction flags.

    Args:
        rows: Processed collision rows.

    Returns:
        ``{name: boolean or Float64 series}``.
    """
    speed = numeric(rows["speed_limit"])
    rural = code_in(rows["urban_or_rural_area"], RURAL)
    high_speed = (speed >= HIGH_SPEED_MPH).astype("boolean")
    junction = rows["junction_detail"]
    is_junction = (junction != NOT_AT_JUNCTION).astype("boolean").mask(junction.isna())
    return {
        "high_speed": high_speed,
        "rural_high_speed": rural & high_speed,
        "rural_single_carriageway_60": rural & code_in(rows["road_type"], SINGLE_CARRIAGEWAY)
        & (speed == HIGH_SPEED_MPH),
        "is_junction": is_junction,
        "junction_complexity": numeric(junction.map(JUNCTION_COMPLEXITY)),
        "uncontrolled_junction": is_junction
        & code_in(rows["junction_control"], GIVE_WAY_UNCONTROLLED),
        "is_roundabout": code_in(rows["road_type"], ROUNDABOUT),
        "is_motorway": code_in(rows["first_road_class"], MOTORWAY_CLASS),
        "crossing_present": (rows["pedestrian_crossing"] != NO_CROSSING).astype("boolean")
        .mask(rows["pedestrian_crossing"].isna()),
        "hazard_present": (rows["carriageway_hazards"] != NO_HAZARD).astype("boolean")
        .mask(rows["carriageway_hazards"].isna()),
    }


# --- vehicle level (order-free aggregation) ---------------------------------------------------
def vehicle_group(vehicles: pd.DataFrame) -> pd.Series:
    """Assign each vehicle row to a group (e-scooter flag overrides vehicle type).

    Args:
        vehicles: Clean vehicle rows with ``vehicle_type`` and ``escooter_flag``.

    Returns:
        String series of group names.
    """
    vtype = vehicles["vehicle_type"]
    group = pd.Series(OTHER_GROUP, index=vehicles.index, dtype=object)
    for name, codes in VEHICLE_GROUPS.items():
        group[vtype.isin(list(codes)).fillna(False).to_numpy()] = name
    group[vehicles["escooter_flag"].eq(1).fillna(False).to_numpy()] = ESCOOTER_GROUP
    return group


def heading_vectors(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Unit travel-direction vectors from compass ``direction_from`` -> ``direction_to``.

    Compass code ``k`` (1 = N ... 8 = NW) sits at bearing ``45 * (k - 1)`` on a unit circle; the
    travel vector is ``point(to) - point(from)``. Parked (0), unknown, or ``from == to``
    (U-turn / no displacement) give NaN.

    Args:
        vehicles: Clean vehicle rows.

    Returns:
        Frame with ``ux`` (east) and ``uy`` (north) float columns.
    """
    def point(codes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        valid = codes.isin(list(COMPASS_CODES)).fillna(False).to_numpy()
        degrees = codes.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)
        bearing = np.deg2rad(45.0 * (degrees - 1))
        bearing[~valid] = np.nan
        return np.sin(bearing), np.cos(bearing)

    fx, fy = point(vehicles["vehicle_direction_from"])
    tx, ty = point(vehicles["vehicle_direction_to"])
    dx, dy = tx - fx, ty - fy
    norm = np.hypot(dx, dy)
    norm[norm < 1e-9] = np.nan
    return pd.DataFrame({"ux": dx / norm, "uy": dy / norm}, index=vehicles.index)


def _any_known(hit: pd.Series, known: pd.Series, key: pd.Series) -> pd.Series:
    """Per-collision "any" flag: 1 if any known row hits, 0 if none, NaN if nothing known."""
    sums = pd.DataFrame({"hit": (hit & known).astype(int), "known": known.astype(int),
                         KEY_COL: key.to_numpy()}).groupby(KEY_COL).sum()
    flag = (sums["hit"] > 0).astype("float64")
    return flag.where(sums["known"] > 0)


def vehicle_level_table(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle derived columns used by the order-free aggregation.

    Args:
        vehicles: Clean vehicle rows.

    Returns:
        Frame aligned with ``vehicles`` (group indicators, mass class, heading, driver fields).
    """
    group = vehicle_group(vehicles)
    mass = vehicles["vehicle_type"].map(MASS_CLASS).astype("float64")
    mass[group.eq(ESCOOTER_GROUP).to_numpy()] = MASS_CLASS[33]
    vru = vehicles["vehicle_type"].isin(list(VRU_TYPES)).fillna(False) | group.eq(ESCOOTER_GROUP)
    table = pd.DataFrame({KEY_COL: vehicles[KEY_COL].astype(str).to_numpy()}, index=vehicles.index)
    for name in [*VEHICLE_GROUPS, OTHER_GROUP]:
        table[f"n_{name}"] = group.eq(name).astype(int).to_numpy()
    table["vru_vehicle"] = vru.astype(int).to_numpy()
    table["mass_class"] = mass.to_numpy()
    table = table.join(heading_vectors(vehicles))
    table["age"] = vehicles["age_of_driver"].astype("float64").to_numpy()
    table["imd"] = vehicles["driver_imd_decile"].astype("float64").to_numpy()
    return table


def manoeuvre_and_driver_flags(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Per-collision "any" flags over manoeuvres, ages, journey purpose and propulsion.

    Args:
        vehicles: Clean vehicle rows.

    Returns:
        Frame indexed by ``collision_index`` (float, NaN when nothing known).
    """
    key = vehicles[KEY_COL].astype(str)
    man = vehicles["vehicle_manoeuvre"]
    age = vehicles["age_of_driver"].astype("Float64")

    def any_of(source: pd.Series, codes: Iterable[int]) -> pd.Series:
        hit = source.isin(list(codes)).fillna(False)
        return _any_known(hit, source.notna(), key)

    return pd.DataFrame({
        "any_overtaking": any_of(man, OVERTAKING),
        "any_right_turn": any_of(man, RIGHT_TURN),
        "any_stationary": any_of(man, STATIONARY),
        "any_reversing": any_of(man, REVERSING),
        "any_young_driver": _any_known((age < YOUNG_AGE).fillna(False), age.notna(), key),
        "any_older_driver": _any_known((age >= OLDER_AGE).fillna(False), age.notna(), key),
        "any_work_journey": any_of(vehicles["journey_purpose_of_driver"], WORK_JOURNEY),
        "any_electric": any_of(vehicles["propulsion_code"], ELECTRIC),
    })


def relative_heading(table: pd.DataFrame, n_rows: pd.Series) -> pd.Series:
    """Angle (0-180 degrees) between the travel directions of two-vehicle collisions.

    Uses ``|u1 + u2|^2 = 2 + 2 cos(theta)`` on the summed unit vectors, which is symmetric in
    the two vehicles (order-free). NaN unless the collision has exactly two vehicle rows with
    defined headings.

    Args:
        table: Output of :func:`vehicle_level_table`.
        n_rows: Vehicle rows per collision (indexed by ``collision_index``).

    Returns:
        Float series indexed by ``collision_index``.
    """
    sums = table.groupby(KEY_COL)[["ux", "uy"]].agg(["sum", "count"])
    n_defined = sums[("ux", "count")]
    sq = sums[("ux", "sum")] ** 2 + sums[("uy", "sum")] ** 2
    cos_theta = np.clip((sq - 2.0) / 2.0, -1.0, 1.0)
    angle = pd.Series(np.rad2deg(np.arccos(cos_theta)), index=sums.index).round(6)
    ok = n_defined.eq(2) & n_rows.reindex(sums.index).eq(2)
    return angle.where(ok)


def vehicle_mix_features(vehicles: pd.DataFrame) -> pd.DataFrame:
    """Order-free collision-level vehicle-mix and driver features.

    Args:
        vehicles: Clean vehicle rows (``vehicles_clean.parquet`` schema).

    Returns:
        Frame indexed by ``collision_index`` with float columns (``has_vru_vehicle`` is later
        combined with the pedestrian flag into ``has_vru``).
    """
    table = vehicle_level_table(vehicles)
    grouped = table.groupby(KEY_COL)
    counts = grouped[[c for c in table.columns if c.startswith("n_")] + ["vru_vehicle"]].sum()
    n_rows = grouped.size()
    sex = vehicles["sex_of_driver"]
    sex_counts = pd.DataFrame({KEY_COL: table[KEY_COL], "male": sex.eq(MALE).fillna(False),
                               "known": sex.isin([MALE, FEMALE]).fillna(False)})
    sex_counts = sex_counts.groupby(KEY_COL)[["male", "known"]].sum()
    out = counts.drop(columns="vru_vehicle").astype("float64")
    out["has_vru_vehicle"] = (counts["vru_vehicle"] > 0).astype("float64")
    out["max_mass_class"] = grouped["mass_class"].max()
    out["min_mass_class"] = grouped["mass_class"].min()
    out["mass_mismatch"] = out["max_mass_class"] - out["min_mass_class"]
    out["relative_heading"] = relative_heading(table, n_rows)
    age_max, age_min = grouped["age"].max(), grouped["age"].min()
    out["driver_age_range"] = age_max - age_min
    out["n_age_unknown"] = (n_rows - grouped["age"].count()).astype("float64")
    out["share_male_driver"] = (sex_counts["male"] / sex_counts["known"]).where(
        sex_counts["known"] > 0)
    out["imd_min"] = grouped["imd"].min()
    out["imd_mean"] = grouped["imd"].mean()
    out = out.join(manoeuvre_and_driver_flags(vehicles))
    logger.info(f"Vehicle-mix features: {out.shape[1]} columns for {len(out):,} collisions")
    return out


def heading_class(angle: pd.Series) -> pd.Categorical:
    """Bucket relative heading into same direction (<=45), head-on (>=135) or crossing."""
    labels = pd.Series(np.where(angle <= 45, "same_direction",
                                np.where(angle >= 135, "head_on", "crossing")), index=angle.index)
    return to_category(labels.where(angle.notna()), HEADING_LEVELS)


# --- pedestrian -------------------------------------------------------------------------------
def pedestrian_flags(rows: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.Categorical]:
    """Child / elderly pedestrian flags and the pedestrian location group.

    Args:
        rows: Frame with ``ped_involved``, ``ped_age``, ``ped_location``.

    Returns:
        ``({name: boolean series}, ped_crossing_group categorical)``.
    """
    involved = rows["ped_involved"].eq(1)
    age = numeric(rows["ped_age"])
    child = (age < CHILD_AGE).astype("boolean").where(involved, False)
    elderly = (age >= OLDER_AGE).astype("boolean").where(involved, False)
    group = rows["ped_location"].map(PED_CROSSING_GROUPS).astype(object)
    group = group.where(involved, NO_PEDESTRIAN)
    return ({"ped_child": child, "ped_elderly": elderly},
            to_category(group, PED_GROUP_LEVELS))


# --- geography --------------------------------------------------------------------------------
def projected_km(rows: pd.DataFrame, config: EngineeredConfig) -> tuple[pd.Series, pd.Series]:
    """Equirectangular projection of lon/lat to km around a fixed GB origin."""
    lon, lat = numeric(rows["longitude"]), numeric(rows["latitude"])
    x = (lon - config.origin_lon) * KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(config.origin_lat))
    y = (lat - config.origin_lat) * KM_PER_DEG_LAT
    return x, y


def grid_cell(rows: pd.DataFrame, config: EngineeredConfig) -> pd.Series:
    """String id of the ``grid_km`` x ``grid_km`` cell containing each collision (NA if no
    coordinates)."""
    x, y = projected_km(rows, config)
    gx = np.floor(x.to_numpy(np.float64, na_value=np.nan) / config.grid_km)
    gy = np.floor(y.to_numpy(np.float64, na_value=np.nan) / config.grid_km)
    ok = ~(np.isnan(gx) | np.isnan(gy))
    cell = np.full(len(rows), None, dtype=object)
    cell[ok] = [f"{int(a)}_{int(b)}" for a, b in zip(gx[ok], gy[ok], strict=True)]
    return pd.Series(cell, index=rows.index)


def rotated_coordinates(rows: pd.DataFrame, config: EngineeredConfig) -> dict[str, pd.Series]:
    """Coordinates rotated by each configured angle (helps axis-aligned tree splits)."""
    x, y = projected_km(rows, config)
    out: dict[str, pd.Series] = {}
    for angle in config.rotation_angles:
        theta = np.deg2rad(angle)
        out[f"rot{angle}_x"] = x * np.cos(theta) + y * np.sin(theta)
        out[f"rot{angle}_y"] = -x * np.sin(theta) + y * np.cos(theta)
    return out


def target_encodings(rows: pd.DataFrame, train: pd.Series, config: EngineeredConfig
                     ) -> pd.DataFrame:
    """Out-of-fold m-estimate target encodings for ONS district, police force and grid cell.

    Training rows get out-of-fold values; all other rows use the full-training-rows map.

    Args:
        rows: Phase rows with ``target``.
        train: Boolean mask of training rows.
        config: Feature configuration.

    Returns:
        float32 frame aligned with ``rows``.
    """
    keys = {
        "te_ons_district": rows["local_authority_ons_district"],
        "te_police_force": rows["police_force"],
        f"te_grid{int(config.grid_km)}km": grid_cell(rows, config),
    }
    y_train = rows.loc[train, "target"].to_numpy()
    out = pd.DataFrame(index=rows.index)
    for name, key in keys.items():
        encoder = OOFTargetEncoder(name=name, m=config.te_m, n_splits=config.te_folds,
                                   seed=config.seed)
        values = pd.Series(np.nan, index=rows.index, dtype=np.float32)
        values[train] = encoder.fit_transform(key[train], y_train).to_numpy()
        values[~train] = encoder.transform(key[~train]).to_numpy()
        out[name] = values.astype(np.float32)
    return out


# --- assembly ---------------------------------------------------------------------------------
def collision_level_features(rows: pd.DataFrame, mix: pd.DataFrame,
                             holidays: BankHolidays, config: EngineeredConfig) -> pd.DataFrame:
    """All label-free engineered features for ``rows`` (no fitted encoders).

    Args:
        rows: Phase rows of the processed collision table.
        mix: Output of :func:`vehicle_mix_features`.
        holidays: Bank-holiday calendars.
        config: Feature configuration.

    Returns:
        Feature frame aligned with ``rows``.
    """
    env = environment_flags(rows)
    road = road_flags(rows)
    veh = mix.reindex(rows[KEY_COL].astype(str).to_numpy()).set_axis(rows.index)
    ped_flags, ped_group = pedestrian_flags(rows)
    ped = rows["ped_involved"].eq(1).astype("boolean")
    has_vru = veh["has_vru_vehicle"].astype("boolean") | ped
    single = (numeric(rows["number_of_vehicles"]) == 1).astype("boolean")
    rural = code_in(rows["urban_or_rural_area"], RURAL)
    flags: dict[str, pd.Series] = {**env, **road, "single_vehicle": single, "has_vru": has_vru,
                                   "vru_vs_heavy": has_vru & (numeric(veh["max_mass_class"])
                                                              >= HEAVY_MASS_CLASS),
                                   **ped_flags}
    flags.update({
        "high_speed_x_vru": road["high_speed"] & has_vru,
        "dark_x_ped": env["is_dark"] & ped,
        "rural_x_single_vehicle": rural & single,
        "speed_x_single_vehicle": numeric(rows["speed_limit"]) * single.astype("Float64"),
    })
    flags.update(rotated_coordinates(rows, config))
    block = _f32_frame(flags, rows.index)
    mix_cols = [c for c in veh.columns if c != "has_vru_vehicle"]
    block = block.join(_f32_frame({c: veh[c] for c in mix_cols}, rows.index))
    block["road_class_pair"] = road_class_pair(rows)
    block["heading_class"] = heading_class(veh["relative_heading"])
    block["ped_crossing_group"] = ped_group
    return block.join(temporal_features(rows, holidays))


def build_f1full(rows: pd.DataFrame, f0: pd.DataFrame, mix: pd.DataFrame, train: pd.Series,
                 config: EngineeredConfig, holidays: BankHolidays | None = None) -> pd.DataFrame:
    """Build the F1-full block: F0 + engineered features + geography target encodings.

    Args:
        rows: Phase rows of the processed collision table (with ``target``).
        f0: F0 block for the same rows (from :func:`src.features.raw.build_f0`).
        mix: Vehicle-mix features (from :func:`vehicle_mix_features`).
        train: Boolean mask of the rows target encoders may be fitted on.
        config: Feature configuration.
        holidays: Bank-holiday calendars (loaded from ``config`` if omitted).

    Returns:
        Feature frame aligned with ``rows``.
    """
    holidays = holidays or load_bank_holidays(config.bank_holidays_path)
    engineered = collision_level_features(rows, mix, holidays, config)
    te = target_encodings(rows, train, config)
    dup = set(f0.columns) & (set(engineered.columns) | set(te.columns))
    if dup:
        raise ValueError(f"Engineered names collide with F0: {sorted(dup)}")
    out = pd.concat([f0, engineered, te], axis=1)
    logger.info(f"F1full block: {out.shape[1]} features = {f0.shape[1]} F0 + "
                f"{engineered.shape[1]} engineered + {te.shape[1]} target encodings")
    return out
