"""Evaluate every frozen cell on the sealed 2025 TEST year and run the RQ contrasts.

Outputs:
    experiments/cells/<cell>/preds_test_seed{s}.parquet
    reports/tables/test_results.csv          (cell x seed x variant x operating point)
    reports/tables/contrasts.csv             (paired bootstrap / DeLong / McNemar / Holm)
    reports/tables/val_test_gap.csv          (CAL vs TEST, mean over seeds)
    reports/tables/test_stratified.csv       (selected model by reporting system / attendance)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path as _Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from joblib import Parallel, delayed

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.split import load_split
from src.evaluation.metrics import expected_calibration_error
from src.evaluation.posthoc import (
    evaluate_frozen,
    load_decision,
    metric_row,
    require_frozen_manifest,
)
from src.evaluation.stats import (
    ThresholdMetric,
    delong_test,
    holm_correction,
    mcnemar_test,
    paired_bootstrap_many,
)
from src.models.registry import predict_proba
from src.paths import EXPERIMENTS_DIR, TABLES_DIR
from src.training.runner import feature_path
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)
CELLS_DIR = EXPERIMENTS_DIR / "cells"
N_BOOT = 2000


# --------------------------------------------------------------------------- integrity
def verify_manifest(manifest: dict[str, Any]) -> None:
    """Recompute provenance hashes and refuse to evaluate if anything changed after the freeze.

    Args:
        manifest: Frozen manifest.

    Raises:
        RuntimeError: On any hash mismatch or missing artefact.
    """
    import hashlib

    from src.paths import CONFIGS_DIR, PROCESSED_DIR, PROJECT_ROOT

    def h(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    roots = {"artefacts": EXPERIMENTS_DIR, "features": PROCESSED_DIR / "features",
             "source": PROJECT_ROOT, "configs": CONFIGS_DIR}
    bad = []
    for group, entries in manifest["hashes"].items():
        if group == "source":
            continue  # evaluation code may be fixed after freeze; models/decisions may not
        for rel, digest in entries.items():
            path = roots[group] / rel
            if not path.exists() or h(path) != digest:
                bad.append(f"{group}/{rel}")
    if bad:
        raise RuntimeError(f"{len(bad)} frozen artefacts changed or missing, e.g. {bad[:5]}")
    logger.info("Manifest verified: %d artefacts unchanged since freeze",
                sum(len(v) for k, v in manifest["hashes"].items() if k != "source"))


# --------------------------------------------------------------------------- predictions
def load_test_frame(feature_set: str) -> pd.DataFrame:
    """Load TEST rows of a refit-phase feature parquet (requires the frozen manifest).

    Args:
        feature_set: Feature-set name.

    Returns:
        TEST rows.
    """
    require_frozen_manifest()
    table = ds.dataset(feature_path(feature_set, "refit")).to_table(
        filter=ds.field("split") == "test")
    return table.to_pandas()


def predict_cell(cell_dir: Path, frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Predict TEST for every seed of a cell and evaluate under the frozen decisions.

    Args:
        cell_dir: Cell directory.
        frames: Cache of TEST frames per feature set.

    Returns:
        Metric rows.
    """
    spec = json.loads((cell_dir / "config.json").read_text())
    fs = spec["feature_set"]
    if fs not in frames:
        frames[fs] = load_test_frame(fs)
    df = frames[fs]
    rows: list[dict[str, Any]] = []
    for model_file in sorted(cell_dir.glob("model_seed*.joblib")):
        seed = int(model_file.stem.split("seed")[-1])
        bundle = joblib.load(model_file)
        adapter = bundle["adapter"]
        X = df[adapter.cat_cols + adapter.num_cols]
        p = predict_proba(bundle["model"], spec["model"], adapter.transform(X), adapter.cat_cols)
        y = df["target"].to_numpy().astype(int)
        pd.DataFrame({"collision_index": df["collision_index"], "target": y, "prob": p}).to_parquet(
            cell_dir / f"preds_test_seed{seed}.parquet", index=False)
        decision, cals = load_decision(cell_dir, seed)
        test_rows, _ = evaluate_frozen(y, p, decision, cals)
        meta = {"cell_id": cell_dir.name, "model": spec["model"], "feature_set": fs,
                "cs": spec["cs"], "tag": spec.get("tag", ""), "seed": seed,
                "best_calibrator": decision.best_calibrator}
        rows.extend(meta | r for r in test_rows)
    return rows


def calibrated_probs(cell_id: str, seed: int, split: str) -> tuple[np.ndarray, dict]:
    """Labels and per-variant probabilities for a cell/seed on CAL or TEST.

    Args:
        cell_id: Cell id.
        seed: Seed.
        split: ``cal`` or ``test``.

    Returns:
        ``(y, {"variants": {variant: p}, "decision": FrozenDecision})``.
    """
    cell_dir = CELLS_DIR / cell_id
    preds = pd.read_parquet(cell_dir / f"preds_{split}_seed{seed}.parquet")
    decision, cals = load_decision(cell_dir, seed)
    p = preds["prob"].to_numpy()
    variants = {v: np.clip(c.predict(p), 0, 1) for v, c in cals.items()}
    return preds["target"].to_numpy().astype(int), {"variants": variants, "decision": decision,
                                                    "ids": preds["collision_index"].to_numpy()}


# --------------------------------------------------------------------------- contrasts
@dataclass
class Arm:
    """One side of a contrast: a cell evaluated with a calibration variant and operating point."""

    cell_id: str
    variant: str
    op: str


@dataclass
class Contrast:
    """A paired comparison A - B on TEST (seed 0)."""

    rq: str
    name: str
    model: str
    a: Arm
    b: Arm
    primary_metric: str
    primary: bool = True


def build_contrasts(cells: set[str], best_cs: dict[str, str], models: list[str]) -> list[Contrast]:
    """Enumerate RQ1-RQ3 and combined contrasts for which both cells exist.

    Args:
        cells: Available cell ids.
        best_cs: CAL-selected CS per ``<feature_set>__<model>``.
        models: Model keys.

    Returns:
        Contrast list.
    """
    out: list[Contrast] = []

    def add(c: Contrast) -> None:
        if c.a.cell_id in cells and c.b.cell_id in cells:
            out.append(c)

    for m in models:
        add(Contrast("RQ1", "F1 vs F0 (no CS, raw)", m, Arm(f"F1__none__{m}", "raw", "tf2"),
                     Arm(f"F0__none__{m}", "raw", "tf2"), "pr_auc"))
        add(Contrast("RQ1", "F1full vs F0 (no CS, raw)", m, Arm(f"F1full__none__{m}", "raw", "tf2"),
                     Arm(f"F0__none__{m}", "raw", "tf2"), "pr_auc", primary=False))
        add(Contrast("RQ1", "F1 (selected) vs F1full", m, Arm(f"F1__none__{m}", "raw", "tf2"),
                     Arm(f"F1full__none__{m}", "raw", "tf2"), "pr_auc", primary=False))
        for fs in ("F0", "F1"):
            for cs in ("weight", "rus", "smotenc"):
                add(Contrast("RQ2", f"{cs} vs none ({fs}, raw)", m,
                             Arm(f"{fs}__{cs}__{m}", "raw", "tf2"),
                             Arm(f"{fs}__none__{m}", "raw", "tf2"), "f2@tf2",
                             primary=(fs == "F0" and cs == "weight")))
        for fs, cs, primary in (("F1", best_cs.get(f"F1__{m}", "none"), True),
                                ("F0", "none", False)):
            cid = f"{fs}__{cs}__{m}"
            tag = f"({fs}, {cs})"
            add(Contrast("RQ3", f"best calibrator vs raw {tag}", m, Arm(cid, "best", "tcost"),
                         Arm(cid, "raw", "tcost"), "brier", primary=primary))
            for v in ("platt", "isotonic"):
                add(Contrast("RQ3", f"{v} vs raw {tag}", m, Arm(cid, v, "tcost"),
                             Arm(cid, "raw", "tcost"), "brier", primary=False))
            add(Contrast("RQ3", f"threshold opt.: raw@t* vs raw@0.5 {tag}", m,
                         Arm(cid, "raw", "tf2"), Arm(cid, "raw", "t050"), "f2@op", primary=False))
            add(Contrast("RQ3", f"calibrated@cost-threshold vs raw@0.5 {tag}", m,
                         Arm(cid, "best", "tcost"), Arm(cid, "raw", "t050"), "cost@op",
                         primary=False))
        cs_star = best_cs.get(f"F1__{m}", "none")
        add(Contrast("ALL", f"F1+{cs_star}+calibrated@tf2 vs F0+none+raw@0.5", m,
                     Arm(f"F1__{cs_star}__{m}", "best", "tf2"),
                     Arm(f"F0__none__{m}", "raw", "t050"),
                     "f2@op", primary=False))
    for tag, label in (("S1", "add Tier-B crash mechanics"), ("S2", "missingness-blind"),
                       ("S3", "drop reporting-system proxies"),
                       ("S5", "drop casualty-derived features")):
        fs = {"S1": "F1S1", "S2": "F1S2"}.get(tag, "F1")
        add(Contrast("SENS", f"{tag}: {label} vs F1", "xgb", Arm(f"{fs}__none__xgb__{tag}", "raw",
                     "tf2"), Arm("F1__none__xgb", "raw", "tf2"), "pr_auc", primary=False))
    return out


def contrast_keys(contrasts: list[Contrast]) -> list[str]:
    """Stable string keys of a contrast family (frozen in the manifest).

    Args:
        contrasts: Contrast list.

    Returns:
        Keys ``rq|model|name|primary``.
    """
    return [f"{c.rq}|{c.model}|{c.name}|{c.primary}" for c in contrasts]


def run_contrast(c: Contrast) -> dict[str, Any]:
    """Paired bootstrap (+ DeLong, McNemar) for one contrast on TEST seed 0.

    Args:
        c: Contrast.

    Returns:
        Flat result rows keyed by metric.
    """
    y, A = calibrated_probs(c.a.cell_id, 0, "test")
    y_b, B = calibrated_probs(c.b.cell_id, 0, "test")
    if not (np.array_equal(A["ids"], B["ids"]) and np.array_equal(y, y_b)):
        raise RuntimeError(f"Arms of '{c.name}' ({c.model}) are not row-aligned on TEST")
    va = A["decision"].best_calibrator if c.a.variant == "best" else c.a.variant
    vb = B["decision"].best_calibrator if c.b.variant == "best" else c.b.variant
    pa, pb = A["variants"][va], B["variants"][vb]
    ta = A["decision"].thresholds[va][c.a.op]
    tb = B["decision"].thresholds[vb][c.b.op]
    t05a, t05b = 0.5, 0.5
    ma = {"pr_auc": "pr_auc", "roc_auc": "roc_auc", "brier": "brier",
          "f2@op": ThresholdMetric("f2", ta), "recall@op": ThresholdMetric("recall", ta),
          "precision@op": ThresholdMetric("precision", ta),
          "cost@op": ThresholdMetric("expected_cost", ta, fn_cost=5.0, fp_cost=1.0),
          "recall@0.5": ThresholdMetric("recall", t05a), "ece": expected_calibration_error}
    mb = dict(ma) | {"f2@op": ThresholdMetric("f2", tb), "recall@op": ThresholdMetric("recall", tb),
                     "precision@op": ThresholdMetric("precision", tb),
                     "cost@op": ThresholdMetric("expected_cost", tb, fn_cost=5.0, fp_cost=1.0),
                     "recall@0.5": ThresholdMetric("recall", t05b)}
    res = paired_bootstrap_many(y, pa, pb, ma, n_boot=N_BOOT, seed=42, metrics_b=mb)
    base = {"rq": c.rq, "contrast": c.name, "model": c.model, "a": f"{c.a.cell_id}|{va}|{c.a.op}",
            "b": f"{c.b.cell_id}|{vb}|{c.b.op}", "thr_a": ta, "thr_b": tb,
            "primary_metric": "f2@op" if c.primary_metric in ("f2@tf2", "f2@op") else
            c.primary_metric, "primary": c.primary}
    dl = delong_test(y, pa, pb)
    mc = mcnemar_test(y, (pa >= ta).astype(int), (pb >= tb).astype(int))
    base |= {"delong_p": dl.p_value, "mcnemar_p": mc.p_value}
    base |= {"ece_a": metric_row(y, pa, ta)["ece"], "ece_b": metric_row(y, pb, tb)["ece"]}
    rows = {}
    for k, r in res.items():
        d = r.to_dict()
        rows |= {f"{k}__a": d["metric_a"], f"{k}__b": d["metric_b"], f"{k}__delta": d["delta"],
                 f"{k}__lo": d["ci_low"], f"{k}__hi": d["ci_high"], f"{k}__p": d["p_value"]}
    return base | rows


def add_holm(df: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjust the primary-metric p-values within the primary family and the combined family.

    Args:
        df: Contrast results.

    Returns:
        Frame with ``p_primary`` and ``p_holm`` columns.
    """
    df = df.copy()
    df["p_primary"] = [r[f"{r['primary_metric']}__p"] for _, r in df.iterrows()]
    df["p_holm"] = np.nan
    fam_primary = df["primary"].astype(bool)
    fam_combined = df["rq"] == "ALL"
    for fam in (fam_primary, fam_combined):
        if fam.any():
            df.loc[fam, "p_holm"] = holm_correction(df.loc[fam, "p_primary"].to_numpy())
    return df


# --------------------------------------------------------------------------- summaries
def val_test_gap(val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """CAL vs TEST metrics (mean over seeds) for each cell/variant/operating point.

    Args:
        val: Validation table.
        test: Test table.

    Returns:
        Merged gap table.
    """
    keys = ["cell_id", "model", "feature_set", "cs", "tag", "variant", "op"]
    metrics = ["pr_auc", "roc_auc", "brier", "ece", "recall", "precision", "f2", "expected_cost"]
    v = val.fillna({"tag": ""}).groupby(keys)[metrics].mean()
    t = test.fillna({"tag": ""}).groupby(keys)[metrics].mean()
    gap = v.join(t, lsuffix="_cal", rsuffix="_test")
    for m in metrics:
        gap[f"{m}_gap"] = gap[f"{m}_test"] - gap[f"{m}_cal"]
    return gap.reset_index()


def stratified(selected: str) -> pd.DataFrame:
    """Selected-model TEST metrics by reporting system and police attendance (seed 0, best cal).

    Args:
        selected: Selected cell id.

    Returns:
        Stratified metric table.
    """
    y, A = calibrated_probs(selected, 0, "test")
    v = A["decision"].best_calibrator
    p, t = A["variants"][v], A["decision"].thresholds[v]["tf2"]
    ids = pd.read_parquet(CELLS_DIR / selected / "preds_test_seed0.parquet")["collision_index"]
    meta = load_split("test", allow_test=True).set_index("collision_index").loc[ids]
    rows = []
    for col in ("collision_injury_based", "police_attended"):
        for level, idx in meta.reset_index().groupby(col).indices.items():
            if len(idx) < 200 or y[idx].min() == y[idx].max():
                continue
            rows.append({"stratifier": col, "level": level, "n": len(idx)}
                        | metric_row(y[idx], p[idx], t))
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()
    setup_logging()
    manifest = require_frozen_manifest()
    verify_manifest(manifest)
    cells = [c for c in manifest["cells"] if c not in set(manifest.get("incomplete_cells", []))
             and (CELLS_DIR / c / "summary.json").exists()]
    if len(cells) < len(manifest["cells"]):
        logger.warning("Evaluating %d of %d frozen cells (manifest marked incomplete)",
                       len(cells), len(manifest["cells"]))
    if not args.skip_predict:
        frames: dict[str, pd.DataFrame] = {}
        rows: list[dict[str, Any]] = []
        for cid in cells:
            rows.extend(predict_cell(CELLS_DIR / cid, frames))
            logger.info("TEST predicted: %s", cid)
        pd.DataFrame(rows).to_csv(TABLES_DIR / "test_results.csv", index=False)
    test = pd.read_csv(TABLES_DIR / "test_results.csv")
    val = pd.read_csv(TABLES_DIR / "validation_results.csv")
    val_test_gap(val, test).to_csv(TABLES_DIR / "val_test_gap.csv", index=False)
    all_models = sorted({c.split("__")[2] for c in manifest["cells"]})
    family = build_contrasts(set(manifest["cells"]), manifest["best_cs"], all_models)
    frozen = manifest.get("contrasts")
    if frozen is not None and frozen != contrast_keys(family):
        raise RuntimeError("Contrast family differs from the one frozen in the manifest")
    available = set(cells)
    contrasts = [c for c in family if c.a.cell_id in available and c.b.cell_id in available]
    logger.info("Running %d contrasts (B=%d)", len(contrasts), N_BOOT)
    results = Parallel(n_jobs=args.n_jobs)(delayed(run_contrast)(c) for c in contrasts)
    add_holm(pd.DataFrame(results)).to_csv(TABLES_DIR / "contrasts.csv", index=False)
    stratified(manifest["selected"]["cell_id"]).to_csv(TABLES_DIR / "test_stratified.csv",
                                                        index=False)
    from src.data.split import split_summary
    from src.paths import COLLISIONS_PATH
    split_summary(pd.read_parquet(COLLISIONS_PATH), include_test_labels=True).to_csv(
        TABLES_DIR / "split_summary.csv", index=False)  # TEST prevalence released post-freeze
    logger.info("TEST evaluation complete")


if __name__ == "__main__":
    main()
