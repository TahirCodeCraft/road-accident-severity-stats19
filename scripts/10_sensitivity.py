"""Sensitivity analyses S1-S4 (plan §4), run on LightGBM with the F1 feature set.

S1  add the borderline "Tier-B" crash-mechanics fields (skidding, hit object, leaving carriageway,
    first point of impact) to F1 -> quantifies how much they would inflate performance.
S2  missingness-blind: numeric NaN -> training median, categorical NaN -> training mode.
S3  drop police_force / te_police_force (reporting-system proxies).
S5  drop every feature derived from the casualty table (pedestrian casualty fields and the
    VRU flags that include them): these record an *injured* pedestrian, so they are borderline.
S4  three-class formulation (Fatal / Serious / Slight) with LightGBM, macro-F1 (post-freeze only).

Usage:
    python scripts/10_sensitivity.py build            # write F1S1 / F1S2 feature parquets
    python scripts/10_sensitivity.py run --model xgb --device cuda  # S1-S3, S5 (pre-freeze)
    python scripts/10_sensitivity.py multiclass       # S4, after scripts/07_freeze.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import multiclass_metrics  # noqa: E402
from src.evaluation.posthoc import require_frozen_manifest  # noqa: E402
from src.paths import COLLISIONS_PATH, EXPERIMENTS_DIR, TABLES_DIR  # noqa: E402
from src.training.runner import ID_COLS, CellSpec, feature_path, run_cell  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)
TRAIN_SPLITS = {"tune": ("fit",), "refit": ("fit", "tune")}
S3_DROP = ["police_force", "te_police_force", "collision_injury_based"]
S5_DROP = ["ped_involved", "ped_age", "ped_location", "ped_movement", "ped_child", "ped_elderly",
           "ped_crossing_group", "dark_x_ped", "has_vru", "vru_vs_heavy"]


def _features(df: pd.DataFrame) -> list[str]:
    """Feature columns of a frame (everything except the ID columns)."""
    return [c for c in df.columns if c not in ID_COLS]


def build_s1(phase: str) -> None:
    """F1 + Tier-B crash-mechanics counts (joined from the processed collision table).

    Args:
        phase: ``tune`` or ``refit``.
    """
    base = pd.read_parquet(feature_path("F1", phase))
    coll = pd.read_parquet(COLLISIONS_PATH)
    tierb = [c for c in coll.columns if c.startswith("tierb_")]
    extra = coll[["collision_index"] + tierb].astype({c: np.float32 for c in tierb})
    out = base.merge(extra, on="collision_index", how="left", validate="one_to_one")
    assert len(out) == len(base)
    out.to_parquet(feature_path("F1S1", phase), index=False)
    logger.info("F1S1_%s: %d features (+%d Tier-B)", phase, len(_features(out)), len(tierb))


def build_s2(phase: str) -> None:
    """F1 with missingness removed using training-row statistics only.

    Args:
        phase: ``tune`` or ``refit``.
    """
    df = pd.read_parquet(feature_path("F1", phase))
    train = df[df["split"].isin(TRAIN_SPLITS[phase])]
    for c in _features(df):
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            mode = train[c].mode(dropna=True)
            if len(mode):
                df[c] = df[c].fillna(mode.iloc[0])
        else:
            df[c] = df[c].fillna(np.float32(train[c].median())).astype(np.float32)
    df.to_parquet(feature_path("F1S2", phase), index=False)
    logger.info("F1S2_%s written (missingness-blind)", phase)


def run_sensitivity_cells(model: str, cs: str, n_trials: int, n_jobs: int, device: str) -> None:
    """Train the S1, S2, S3 and S5 cells (compare each with ``F1__<cs>__<model>``).

    Args:
        model: Registry key of the model family.
        cs: Imbalance strategy of the reference cell.
        n_trials: Optuna trials.
        n_jobs: Threads.
        device: ``cpu`` or ``cuda``.
    """
    common = {"n_trials": n_trials, "n_jobs": n_jobs, "device": device}
    specs = [
        CellSpec(model, "F1S1", cs, tag="S1", **common),
        CellSpec(model, "F1S2", cs, tag="S2", **common),
        CellSpec(model, "F1", cs, tag="S3", drop_features=S3_DROP, **common),
        CellSpec(model, "F1", cs, tag="S5", drop_features=S5_DROP, **common),
    ]
    for s in specs:
        run_cell(s)


def run_multiclass(seed: int = 0) -> None:
    """S4: three-class LightGBM with the frozen F1 LightGBM hyperparameters.

    Trained on FIT+TUNE, decisions by argmax of class-balanced probabilities (fixed a priori),
    evaluated on CAL and on TEST (after freeze).

    Args:
        seed: Random seed.
    """
    from lightgbm import LGBMClassifier

    from src.features.adapters import make_adapter

    manifest = require_frozen_manifest()
    cs = manifest["best_cs"].get("F1__lgbm", "none")
    cell = EXPERIMENTS_DIR / "cells" / f"F1__{cs}__lgbm"
    best = json.loads((cell / "best_params.json").read_text())
    n_est = int(round(1.2 * (best["best_iteration"] or 500)))
    set_seed(seed)
    df = pd.read_parquet(feature_path("F1", "refit"))
    sev = pd.read_parquet(COLLISIONS_PATH, columns=["collision_index", "severity3"])
    df = df.merge(sev, on="collision_index", how="left", validate="one_to_one")
    y3 = (df["severity3"].astype(int) - 1).to_numpy()  # 0 fatal, 1 serious, 2 slight
    feats = [c for c in _features(df) if c != "severity3"]
    adapter = make_adapter("lgbm")
    tr = df["split"].isin(["fit", "tune"]).to_numpy()
    X_tr = adapter.fit_transform(df.loc[tr, feats])
    model = LGBMClassifier(objective="multiclass", n_estimators=n_est, class_weight="balanced",
                           random_state=seed, verbose=-1, n_jobs=8, **best["params"])
    model.fit(X_tr, y3[tr])
    rows = []
    for split in ("cal", "test"):
        m = (df["split"] == split).to_numpy()
        pred = model.predict(adapter.transform(df.loc[m, feats]))
        res = multiclass_metrics(y3[m], pred, labels=[0, 1, 2])
        row = {"split": split, "n": int(m.sum()), "macro_f1": res.macro_f1,
               "balanced_accuracy": res.balanced_accuracy, "accuracy": res.accuracy}
        for k, name in enumerate(("fatal", "serious", "slight")):
            row[f"recall_{name}"] = float(res.per_class_recall[k])
            row[f"precision_{name}"] = float(res.per_class_precision[k])
            row[f"f1_{name}"] = float(res.per_class_f1[k])
        rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "sensitivity_s4_multiclass.csv", index=False)
    logger.info("S4 multiclass: %s", rows)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "run", "multiclass"])
    parser.add_argument("--cs", default="none")
    parser.add_argument("--model", default="xgb")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    setup_logging()
    if args.action == "build":
        for phase in ("tune", "refit"):
            build_s1(phase)
            build_s2(phase)
    elif args.action == "run":
        run_sensitivity_cells(args.model, args.cs, args.n_trials, args.n_jobs, args.device)
    else:
        run_multiclass()


if __name__ == "__main__":
    main()
