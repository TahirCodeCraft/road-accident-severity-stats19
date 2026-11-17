"""Build every result figure and summary table used by the report (run after 08 and 09).

Outputs: reports/figures/fig_*.png and reports/tables/{main_results, ablation_test,
rq_contrasts_primary, sensitivity}.csv (+ .md copies for quick reading).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation import report_plots as rp  # noqa: E402
from src.evaluation.posthoc import load_decision, require_frozen_manifest  # noqa: E402
from src.paths import EXPERIMENTS_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)
CELLS = EXPERIMENTS_DIR / "cells"
KEY_METRICS = ["pr_auc", "roc_auc", "brier", "ece", "recall", "precision", "f2", "expected_cost"]


def write_table(df: pd.DataFrame, name: str) -> None:
    """Write a table as CSV and Markdown.

    Args:
        df: Table.
        name: Base file name.
    """
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    (TABLES_DIR / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))


def seed_summary(test: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over seeds for each untagged cell / variant / operating point.

    Args:
        test: ``test_results.csv``.

    Returns:
        Aggregated table.
    """
    t = test[test["tag"].fillna("") == ""]
    keys = ["cell_id", "model", "feature_set", "cs", "variant", "op"]
    agg = t.groupby(keys)[KEY_METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def main_results(test: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Best CAL configuration per model family on TEST (F1, CAL-best CS, best calibrator, t*).

    Args:
        test: ``test_results.csv``.
        manifest: Frozen manifest.

    Returns:
        One row per model.
    """
    rows = []
    for m in rp.MODEL_ORDER:
        cs = manifest["best_cs"].get(f"F1__{m}")
        if cs is None:
            continue
        g = test[(test["cell_id"] == f"F1__{cs}__{m}") & (test["op"] == "tf2")]
        g = g[g["variant"] == g["best_calibrator"]]
        if g.empty:
            continue
        row = {"model": rp.MODEL_LABELS[m], "config": f"F1 + {rp.CS_LABELS[cs]}",
               "calibrator": g["best_calibrator"].mode().iloc[0]}
        for k in KEY_METRICS:
            row[k] = f"{g[k].mean():.4f} ± {g[k].std():.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def label_contrast(df: pd.DataFrame) -> pd.Series:
    """Readable labels ``Model — contrast``.

    Args:
        df: Contrast rows.

    Returns:
        Label series.
    """
    return df["model"].map(rp.MODEL_LABELS).fillna(df["model"]) + " — " + df["contrast"]


def probs_for(cell_id: str, split: str, seed: int = 0) -> tuple[np.ndarray, dict, object]:
    """Labels, per-variant probabilities and frozen decision for one cell/seed.

    Args:
        cell_id: Cell id.
        split: ``cal`` or ``test``.
        seed: Seed.

    Returns:
        ``(y, {variant: p}, decision)``.
    """
    preds = pd.read_parquet(CELLS / cell_id / f"preds_{split}_seed{seed}.parquet")
    decision, cals = load_decision(CELLS / cell_id, seed)
    p = preds["prob"].to_numpy()
    return (preds["target"].to_numpy().astype(int),
            {v: np.clip(c.predict(p), 0, 1) for v, c in cals.items()}, decision)


def main() -> None:
    """Entry point."""
    setup_logging()
    manifest = require_frozen_manifest()
    test = pd.read_csv(TABLES_DIR / "test_results.csv")
    contrasts = pd.read_csv(TABLES_DIR / "contrasts.csv")
    gap = pd.read_csv(TABLES_DIR / "val_test_gap.csv")
    sel = manifest["selected"]["cell_id"]

    rp.pipeline_diagram(FIGURES_DIR / "fig_01_pipeline.png")
    write_table(main_results(test, manifest), "main_results")
    summ = seed_summary(test)
    write_table(summ, "ablation_test_seed_summary")

    raw = summ[(summ["variant"] == "raw") & (summ["op"] == "t050")].rename(
        columns={f"{k}_mean": k for k in KEY_METRICS})
    rp.ablation_heatmap(raw[raw["feature_set"].isin(["F0", "F1"])], "pr_auc",
                        FIGURES_DIR / "fig_02_ablation_prauc.png",
                        "PR-AUC by model, feature set and imbalance strategy")
    best = summ.copy()
    best = best[(best["op"] == "tf2")]
    cal_best = test.groupby("cell_id")["best_calibrator"].agg(lambda s: s.mode().iloc[0])
    best = best[best["variant"] == best["cell_id"].map(cal_best)].rename(
        columns={f"{k}_mean": k for k in KEY_METRICS})
    rp.ablation_heatmap(best[best["feature_set"].isin(["F0", "F1"])], "f2",
                        FIGURES_DIR / "fig_03_ablation_f2.png",
                        "F2 at the CAL-chosen threshold (calibrated) by configuration")

    c = contrasts.copy()
    c["label"] = label_contrast(c)
    rq1 = c[(c["rq"] == "RQ1")]
    rp.forest_plot(rq1, "pr_auc", FIGURES_DIR / "fig_04_rq1_prauc.png",
                   "RQ1 — engineered + selected features vs raw features",
                   "Difference in PR-AUC on 2025 (A − B)", "Δ PR-AUC")
    rq2 = c[(c["rq"] == "RQ2") & c["contrast"].str.contains("(F0", regex=False)]
    rp.forest_plot(rq2, "recall@0.5", FIGURES_DIR / "fig_05_rq2_recall05.png",
                   "RQ2 — cost-sensitive learning vs none: recall at 0.5 (F0 features)",
                   "The classic claim: more severe cases flagged at the default cut-off",
                   "Δ recall@0.5")
    rp.forest_plot(rq2, "f2@op", FIGURES_DIR / "fig_06_rq2_f2.png",
                   "RQ2 — cost-sensitive learning vs none: F2 at each arm's own t* (F0)",
                   "Once thresholds are tuned on CAL, is there any gain left?", "Δ F2")
    rq3 = c[(c["rq"] == "RQ3") & c["contrast"].str.startswith("best calibrator vs raw")]
    rp.forest_plot(rq3, "brier", FIGURES_DIR / "fig_07_rq3_brier.png",
                   "RQ3 — calibration vs uncalibrated: Brier score",
                   "Negative = better calibrated probabilities on 2025", "Δ Brier")
    rq3c = c[(c["rq"] == "RQ3") & c["contrast"].str.startswith("calibrated@cost")]
    rp.forest_plot(rq3c, "cost@op", FIGURES_DIR / "fig_08_rq3_cost.png",
                   "RQ3 — calibrated cost threshold (1/6) vs raw@0.5: expected cost",
                   "FN:FP cost 5:1, per collision; negative = cheaper", "Δ expected cost")
    comb = c[c["rq"] == "ALL"]
    rp.forest_plot(comb, "f2@op", FIGURES_DIR / "fig_09_combined_f2.png",
                   "All three techniques vs plain baseline",
                   "F1 + CAL-best CS + calibrated@t* vs F0 + none + raw@0.5", "Δ F2")
    sens = c[c["rq"] == "SENS"]
    if not sens.empty:
        rp.forest_plot(sens, "pr_auc", FIGURES_DIR / "fig_10_sensitivity.png",
                       "Sensitivity analyses (XGBoost, F1, no CS)",
                       "PR-AUC change vs the primary F1 model", "Δ PR-AUC")
        write_table(sens[["contrast", "pr_auc__a", "pr_auc__b", "pr_auc__delta", "pr_auc__lo",
                          "pr_auc__hi", "roc_auc__delta", "brier__delta"]], "sensitivity")

    prim = c[c["primary"].astype(bool) | (c["rq"] == "ALL")]
    cols = ["rq", "model", "contrast", "primary_metric", "p_primary", "p_holm"]
    out = prim[cols].copy()
    out["delta"] = [r[f"{r['primary_metric']}__delta"] for _, r in prim.iterrows()]
    out["ci"] = [f"[{r[r['primary_metric'] + '__lo']:+.4f}, {r[r['primary_metric'] + '__hi']:+.4f}]"
                 for _, r in prim.iterrows()]
    write_table(out, "rq_contrasts_primary")

    y, probs, dec = probs_for(sel, "test")
    rp.reliability_panel(y, probs, FIGURES_DIR / "fig_11_reliability_selected.png",
                         "Reliability on 2025 — selected model",
                         f"{sel}: calibrators fitted on CAL 2024, applied unchanged")
    v = dec.best_calibrator
    rp.threshold_curves(y, probs[v], {k: t for k, t in dec.thresholds[v].items()
                                      if k in ("t050", "tf2", "trec80", "tcost")},
                        FIGURES_DIR / "fig_12_threshold_curves.png",
                        "Operating points frozen on CAL, evaluated on 2025",
                        f"Selected model, {v} calibration")
    rp.confusion_pair(y, probs[v], {"Default 0.5": 0.5, "F2-optimal t*": dec.thresholds[v]["tf2"]},
                      FIGURES_DIR / "fig_13_confusion.png")
    curves = {}
    for m in rp.MODEL_ORDER:
        cs = manifest["best_cs"].get(f"F1__{m}")
        if cs and (CELLS / f"F1__{cs}__{m}" / "preds_test_seed0.parquet").exists():
            ym, pm, _ = probs_for(f"F1__{cs}__{m}", "test")
            curves[f"{rp.MODEL_LABELS[m]} ({rp.CS_LABELS[cs]})"] = pm["raw"]
    rp.pr_roc_curves(y, curves, FIGURES_DIR / "fig_14_pr_roc.png")
    g = gap[(gap["variant"] == "raw") & (gap["op"] == "t050") & (gap["tag"].fillna("") == "")]
    rp.val_test_scatter(g, "pr_auc", FIGURES_DIR / "fig_15_val_test_prauc.png")
    strat_path = TABLES_DIR / "test_stratified.csv"
    if strat_path.exists():
        rp.stratified_bars(pd.read_csv(strat_path), FIGURES_DIR / "fig_16_stratified.png")
    (TABLES_DIR / "selected_model.json").write_text(json.dumps(manifest["selected"], indent=2))
    logger.info("Report assets written")


if __name__ == "__main__":
    main()
