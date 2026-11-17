"""Teaching demo: build five engineered features by hand from the downloaded STATS19 CSVs.

Rebuilds ``has_vru``, ``n_age_unknown``, ``min_mass_class``, ``te_ons_district`` and
``relative_heading`` step by step for a few 2023 collisions, then checks each value against the
project's feature table (``F1full_refit.parquet``). Nothing here is used by the pipeline.

Usage:
    python scripts/demo_five_features.py
    python scripts/demo_five_features.py --collision 2023170M10633
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.encoders import OOFTargetEncoder  # noqa: E402
from src.features.engineered import MASS_CLASS, VRU_TYPES  # noqa: E402
from src.paths import COLLISIONS_PATH, PROCESSED_DIR, raw_csv_path  # noqa: E402

logger = logging.getLogger("demo")

EXAMPLES = ("2023170M10633", "2023332300352", "2023170S11103", "2023161291452", "2023161352998")
COMPASS = {1: "N", 2: "NE", 3: "E", 4: "SE", 5: "S", 6: "SW", 7: "W", 8: "NW"}
VEHICLE_NAMES = {1: "pedal cycle", 2: "motorcycle ≤50cc", 3: "motorcycle 50–125cc",
                 4: "motorcycle 125–500cc", 5: "motorcycle >500cc", 8: "taxi", 9: "car",
                 10: "minibus", 11: "bus/coach", 19: "van ≤3.5t", 20: "goods 3.5–7.5t",
                 21: "goods ≥7.5t", 23: "e-motorcycle", 33: "e-scooter", 90: "other",
                 97: "motorcycle (cc unknown)", 98: "goods (weight unknown)"}
PEDESTRIAN_CLASS = 3
TE_M, TE_FOLDS, TE_SEED = 50.0, 5, 42


@dataclass(frozen=True)
class RawTables:
    """Raw 2023 vehicle and casualty rows (all strings, exactly as downloaded)."""

    vehicles: pd.DataFrame
    casualties: pd.DataFrame


def as_int(value: str) -> int | None:
    """Parse a raw code; negative codes (e.g. ``-1``) mean missing.

    Args:
        value: Raw CSV string.

    Returns:
        Integer code or ``None`` when missing.
    """
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return None
    return None if code < 0 else code


def load_raw(year: int = 2023) -> RawTables:
    """Read the downloaded vehicle and casualty CSVs as strings.

    Args:
        year: Data year.

    Returns:
        Raw tables.
    """
    return RawTables(vehicles=pd.read_csv(raw_csv_path("vehicle", year), dtype=str),
                     casualties=pd.read_csv(raw_csv_path("casualty", year), dtype=str))


def build_has_vru(veh: pd.DataFrame, cas: pd.DataFrame) -> tuple[float, list[str]]:
    """1 if any vehicle is a vulnerable road user type or any casualty is a pedestrian."""
    steps, hit = [], False
    for _, v in veh.iterrows():
        code = as_int(v.vehicle_type)
        is_vru = code in VRU_TYPES or as_int(v.escooter_flag) == 1
        hit |= is_vru
        steps.append(f"vehicle {v.vehicle_reference}: type {code} "
                     f"({VEHICLE_NAMES.get(code, 'other/unknown')}) -> VRU? {'yes' if is_vru else 'no'}")
    peds = int((cas.casualty_class.map(as_int) == PEDESTRIAN_CLASS).sum())
    steps.append(f"pedestrian casualties in casualty CSV: {peds}")
    return float(hit or peds > 0), steps


def build_n_age_unknown(veh: pd.DataFrame) -> tuple[float, list[str]]:
    """Number of vehicle rows whose driver age is missing (``-1``)."""
    ages = [as_int(a) for a in veh.age_of_driver]
    steps = [f"driver ages: {[a if a is not None else 'missing' for a in ages]}"]
    return float(sum(a is None for a in ages)), steps


def build_min_mass_class(veh: pd.DataFrame) -> tuple[float, list[str]]:
    """Lightest vehicle's mass class (1 cycle, 2 motorcycle, 3 car, 4 van, 5 medium, 6 heavy)."""
    classes, steps = [], []
    for _, v in veh.iterrows():
        code = as_int(v.vehicle_type)
        mass = 1 if as_int(v.escooter_flag) == 1 else MASS_CLASS.get(code)
        classes.append(mass)
        steps.append(f"vehicle {v.vehicle_reference}: type {code} -> mass class {mass}")
    known = [c for c in classes if c is not None]
    return (float(min(known)) if known else math.nan), steps


def unit_vector(from_code: int | None, to_code: int | None) -> np.ndarray | None:
    """Unit travel direction from compass codes (1 N … 8 NW); ``None`` if undefined."""
    if from_code not in COMPASS or to_code not in COMPASS:
        return None

    def point(code: int) -> np.ndarray:
        bearing = math.radians(45 * (code - 1))
        return np.array([math.sin(bearing), math.cos(bearing)])

    d = point(to_code) - point(from_code)
    norm = float(np.hypot(*d))
    return None if norm < 1e-9 else d / norm


def build_relative_heading(veh: pd.DataFrame) -> tuple[float, list[str]]:
    """Angle between the two vehicles' travel directions (two-vehicle collisions only)."""
    steps, vecs = [], []
    for _, v in veh.iterrows():
        f, t = as_int(v.vehicle_direction_from), as_int(v.vehicle_direction_to)
        u = unit_vector(f, t)
        vecs.append(u)
        label = f"{COMPASS.get(f, f)} -> {COMPASS.get(t, t)}"
        steps.append(f"vehicle {v.vehicle_reference}: {label}, direction vector "
                     f"{'undefined' if u is None else f'({u[0]:+.3f}, {u[1]:+.3f})'}")
    if len(vecs) != 2 or any(u is None for u in vecs):
        steps.append("needs exactly two vehicles with known directions -> NaN")
        return math.nan, steps
    cos = float(np.clip(vecs[0] @ vecs[1], -1, 1))
    angle = math.degrees(math.acos(cos))
    steps.append(f"cos(angle) = {cos:+.3f} -> angle = {angle:.0f}°")
    return round(angle, 6), steps


def district_encoding() -> pd.DataFrame:
    """Per-row te_ons_district for training rows (FIT + TUNE), plus the plain full-map pieces.

    Returns:
        Frame indexed by ``collision_index`` with ``district``, ``n``, ``severe``, ``prior``,
        ``full_map`` and ``oof``.
    """
    rows = pd.read_parquet(COLLISIONS_PATH, columns=["collision_index", "target",
                                                    "local_authority_ons_district"])
    split = pd.read_parquet(PROCESSED_DIR / "features" / "F1full_refit.parquet",
                            columns=["collision_index", "split"])
    rows = rows.merge(split, on="collision_index")
    train = rows[rows.split.isin(["fit", "tune"])].reset_index(drop=True)
    key, y = train["local_authority_ons_district"], train["target"].to_numpy()
    stats = train.groupby("local_authority_ons_district")["target"].agg(["size", "sum"])
    prior = float(y.mean())
    enc = OOFTargetEncoder(name="te_ons_district", m=TE_M, n_splits=TE_FOLDS, seed=TE_SEED)
    out = pd.DataFrame({"collision_index": train.collision_index, "district": key,
                        "oof": enc.fit_transform(key, y).to_numpy()})
    out["n"] = key.map(stats["size"]).to_numpy()
    out["severe"] = key.map(stats["sum"]).to_numpy()
    out["prior"] = prior
    out["full_map"] = (out.severe + TE_M * prior) / (out.n + TE_M)
    return out.set_index("collision_index")


def explain(cid: str, raw: RawTables, te: pd.DataFrame, project: pd.DataFrame) -> list[dict]:
    """Log the step-by-step build of all five features for one collision.

    Args:
        cid: Collision index.
        raw: Raw 2023 tables.
        te: Output of :func:`district_encoding`.
        project: Project feature table indexed by ``collision_index``.

    Returns:
        One summary dict per feature.
    """
    veh = raw.vehicles[raw.vehicles.collision_index == cid].sort_values("vehicle_reference")
    cas = raw.casualties[raw.casualties.collision_index == cid]
    logger.info(f"\n================ Collision {cid} ({len(veh)} vehicles, {len(cas)} casualties)")
    built: dict[str, tuple[float, list[str]]] = {
        "has_vru": build_has_vru(veh, cas),
        "n_age_unknown": build_n_age_unknown(veh),
        "min_mass_class": build_min_mass_class(veh),
        "relative_heading": build_relative_heading(veh),
    }
    r = te.loc[cid]
    built["te_ons_district"] = (float(r.oof), [
        f"district {r.district}: {int(r.n):,} training collisions, {int(r.severe):,} severe "
        f"(own rate {r.severe / r.n:.3f}); national prior {r.prior:.3f}",
        f"smoothed = ({int(r.severe)} + 50 × {r.prior:.3f}) / ({int(r.n)} + 50) = {r.full_map:.3f}",
        f"out-of-fold value (this row's own label left out, 5 folds) = {r.oof:.3f}"])
    summary = []
    for name in ("has_vru", "n_age_unknown", "min_mass_class", "te_ons_district",
                 "relative_heading"):
        value, steps = built[name]
        expected = float(project.loc[cid, name])
        both_nan = math.isnan(value) and math.isnan(expected)
        match = both_nan or (not math.isnan(value) and abs(value - expected) < 1e-3)
        logger.info(f"\n  {name}")
        for s in steps:
            logger.info(f"    - {s}")
        logger.info(f"    => demo {value:.3f} | project {expected:.3f} | "
                    f"{'MATCH' if match else 'DIFFERENT'}")
        summary.append({"collision": cid, "feature": name, "demo": round(value, 3),
                        "project": round(expected, 3), "match": match})
    return summary


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collision", nargs="*", default=list(EXAMPLES),
                        help="2023 collision_index values to rebuild")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = load_raw()
    te = district_encoding()
    project = pd.read_parquet(PROCESSED_DIR / "features" / "F1full_refit.parquet").set_index(
        "collision_index")
    rows = [s for cid in args.collision for s in explain(cid, raw, te, project)]
    table = pd.DataFrame(rows).pivot(index="collision", columns="feature", values="demo")
    ok = pd.DataFrame(rows)["match"]
    logger.info(f"\n================ Summary (demo values)\n{table.to_string()}")
    logger.info(f"\n{int(ok.sum())} of {len(ok)} values match the project's feature table.")


if __name__ == "__main__":
    main()
