"""Build the vector figures (and the numbers quoted with them) for the IEEE paper.

Everything is recomputed from saved artefacts: ``reports/tables/test_results.csv``, the 2025
prediction files and frozen calibrators/thresholds in ``experiments/cells/``, and
``reports/tables/shap_importance.csv``. Nothing is refitted.

Usage:
    python scripts/14_paper_figures.py --out paper/figures
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.thresholds import min_cost_threshold  # noqa: E402

logger = logging.getLogger("paper_figures")

CELLS = _ROOT / "experiments" / "cells"
TABLES = _ROOT / "reports" / "tables"
COL_W, DBL_W = 3.5, 7.16
FN_COST, FP_COST = 5.0, 1.0
N_BINS = 15
MODELS = ("lr", "dt", "rf", "xgb", "lgbm", "cat")
MODEL_LABEL = {"lr": "LR", "dt": "DT", "rf": "RF", "xgb": "XGB", "lgbm": "LGBM", "cat": "CatB"}
C_BLUE, C_AMBER, C_GREEN, C_RED, C_GREY = "#1b4f9c", "#d18f00", "#0a6a44", "#b3261e", "#6b7280"
SHAP_LABEL = {
    "has_vru": "Vulnerable road user involved",
    "police_force": "Police force",
    "n_age_unknown": "Drivers of unknown age",
    "speed_limit": "Speed limit",
    "min_mass_class": "Lightest vehicle mass class",
    "te_ons_district": "District target encoding",
    "relative_heading": "Relative heading (2 vehicles)",
    "veh_vehicle_manoeuvre__19": "Vehicles going ahead",
    "veh_driver_imd_decile__missing": "Driver IMD decile missing",
    "veh_vehicle_left_hand_drive__missing": "Left-hand-drive flag missing",
    "ped_age": "Pedestrian age",
    "hour_cos": "Hour of day (cosine)",
}


@dataclass(frozen=True)
class CellPreds:
    """2025 labels, raw scores and calibrated scores of one ablation cell (seed 0)."""

    y: np.ndarray
    raw: np.ndarray
    platt: np.ndarray
    isotonic: np.ndarray
    platt_coef: tuple[float, float]


def set_style() -> None:
    """Serif, small-font style that matches IEEE two-column text."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def load_cell(cell: str) -> CellPreds:
    """Load seed-0 2025 predictions and apply the CAL-fitted calibrators.

    Args:
        cell: Ablation cell id, e.g. ``F1__weight__lgbm``.

    Returns:
        Labels and raw/calibrated probabilities.
    """
    d = pd.read_parquet(CELLS / cell / "preds_test_seed0.parquet")
    y, p = d["target"].to_numpy(), d["prob"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        platt = joblib.load(CELLS / cell / "calibrator_platt_seed0.joblib")
        iso = joblib.load(CELLS / cell / "calibrator_isotonic_seed0.joblib")
    return CellPreds(y, p, platt.predict(p), iso.predict(p), platt.coef)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple[np.ndarray, np.ndarray, float]:
    """Equal-mass reliability bins and ECE (same definition as the evaluation code).

    Args:
        y: Binary labels.
        p: Probabilities.
        n_bins: Number of equal-mass bins.

    Returns:
        Mean prediction per bin, observed rate per bin, ECE.
    """
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    mp, ob, w = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mp.append(p[m].mean()); ob.append(y[m].mean()); w.append(m.sum())
    mp_a, ob_a, w_a = np.array(mp), np.array(ob), np.array(w)
    return mp_a, ob_a, float(np.sum(np.abs(mp_a - ob_a) * w_a) / w_a.sum())


def curve(y: np.ndarray, p: np.ndarray, ts: np.ndarray) -> pd.DataFrame:
    """Recall, precision, flagged share and expected cost at each threshold (predict p >= t).

    Args:
        y: Binary labels.
        p: Probabilities.
        ts: Thresholds.

    Returns:
        One row per threshold.
    """
    order = np.sort(p)
    pos = y.sum()
    rows = []
    for t in ts:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int(pred.sum() - tp); fn = int(pos - tp)
        rows.append({"t": t, "recall": tp / pos, "precision": tp / max(pred.sum(), 1),
                     "flagged": pred.mean(), "cost": (FN_COST * fn + FP_COST * fp) / len(y)})
    del order
    return pd.DataFrame(rows)


def exact_point(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    """Recall, precision, flagged share and 5:1 expected cost at one exact threshold (p >= t).

    Args:
        y: Binary labels.
        p: Probabilities.
        t: Threshold.

    Returns:
        Metrics at ``t``.
    """
    row = curve(y, p, np.array([t])).iloc[0]
    return {k: float(row[k]) for k in ("recall", "precision", "flagged", "cost")}


def fig_prauc(out: Path) -> dict:
    """PR-AUC on 2025 per model family and configuration (mean of three seeds)."""
    d = pd.read_csv(TABLES / "test_results.csv")
    d = d[(d.variant == "raw") & (d.op == "t050") & d["tag"].isna()]  # exclude sensitivity cells
    g = d.groupby(["model", "feature_set", "cs"]).pr_auc.mean()
    arms = [("F0", "none", "F0, none", "o", "white", C_GREY), ("F1", "none", "F1, none", "o", C_BLUE, C_BLUE),
            ("F1", "weight", "F1, class weight", "^", C_AMBER, C_AMBER),
            ("F1", "rus", "F1, undersampling", "s", C_GREEN, C_GREEN),
            ("F1", "smotenc", "F1, SMOTE-NC", "X", C_RED, C_RED)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    offs = np.linspace(-0.24, 0.24, len(arms))
    for k, (fs, cs, lab, mk, fc, ec) in enumerate(arms):
        xs, ys = [], []
        for i, m in enumerate(MODELS):
            if (m, fs, cs) in g.index:
                xs.append(i + offs[k]); ys.append(g[(m, fs, cs)])
        ax.scatter(xs, ys, marker=mk, s=22, facecolor=fc, edgecolor=ec, linewidth=0.9, label=lab, zorder=3)
    ax.set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel("PR-AUC on 2025")
    ax.set_ylim(0.40, 0.505)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(ncol=2, frameon=False, loc="lower right", handletextpad=0.2, columnspacing=0.8)
    fig.savefig(out / "fig_prauc.pdf"); plt.close(fig)
    return {f"{m}|{fs}|{cs}": round(float(v), 4) for (m, fs, cs), v in g.items()}


def fig_cs_cal(out: Path) -> dict:
    """(a) weighting as a threshold shift, (b) reliability, (c) Platt map vs prior correction."""
    none, wgt = load_cell("F1__none__lgbm"), load_cell("F1__weight__lgbm")
    ft = pd.read_csv(TABLES / "split_summary.csv").set_index("split").loc[["fit", "tune"]]
    w = float((ft.rows.sum() - ft.n_severe.sum()) / ft.n_severe.sum())
    ts = np.round(np.arange(0.02, 0.981, 0.01), 2)
    cn, cw = curve(none.y, none.raw, ts), curve(wgt.y, wgt.raw, ts)
    t_eq = 0.5 / (w - w * 0.5 + 0.5)
    at = lambda c, t: {k: float(np.interp(t, c.t, c[k])) for k in ("recall", "precision", "flagged")}  # noqa: E731
    eq_none, eq_w = exact_point(none.y, none.raw, t_eq), exact_point(wgt.y, wgt.raw, 0.5)
    del at

    fig, axs = plt.subplots(1, 3, figsize=(DBL_W, 2.05))
    a = axs[0]
    a.plot(cn.t, cn.recall, color=C_BLUE, label="no weighting")
    a.plot(cw.t, cw.recall, color=C_AMBER, label=f"class weight $w$={w:.2f}")
    a.plot([t_eq, 0.5], [eq_none["recall"], eq_w["recall"]], ls=":", color=C_GREY, lw=1)
    a.scatter([t_eq, 0.5], [eq_none["recall"], eq_w["recall"]], color=[C_BLUE, C_AMBER], s=16, zorder=3)
    a.annotate(f"$t$={t_eq:.2f}", (t_eq, eq_none["recall"]), xytext=(-26, -20), textcoords="offset points", fontsize=8)
    a.annotate("$t$=0.50", (0.5, eq_w["recall"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    a.set_xlabel("decision threshold"); a.set_ylabel("recall on 2025"); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    a.legend(frameon=False, loc="lower left", handlelength=1.2); a.set_title("(a) Weighting moves the operating point", loc="left")

    b = axs[1]
    b.plot([0, 1], [0, 1], ls="--", color=C_GREY, lw=0.8)
    out_ece = {}
    for lab, p, col, mk in [("weighted, raw", wgt.raw, C_AMBER, "^"), ("weighted, Platt", wgt.platt, C_BLUE, "o"),
                            ("weighted, isotonic", wgt.isotonic, C_GREEN, "s")]:
        mp, ob, e = reliability(wgt.y, p)
        out_ece[lab] = round(e, 4)
        b.plot(mp, ob, marker=mk, ms=3, color=col, label=f"{lab} (ECE {e:.3f})")
    b.set_xlabel("mean predicted risk (15 quantile bins)"); b.set_ylabel("observed severe share")
    b.set_xlim(0, 0.9); b.set_ylim(0, 0.9); b.legend(frameon=False, loc="upper left", handlelength=1.0, fontsize=8, borderaxespad=0.2)
    b.set_title("(b) Reliability on 2025", loc="left")

    c = axs[2]
    grid = np.linspace(0.005, 0.995, 300)
    lg = np.log(grid / (1 - grid))
    sa, sb = wgt.platt_coef
    c.plot([0, 1], [0, 1], ls="--", color=C_GREY, lw=0.8)
    c.plot(grid, 1 / (1 + np.exp(-(sa * lg + sb))), color=C_BLUE, label=f"fitted Platt ($a$={sa:.2f}, $b$={sb:.2f})")
    c.plot(grid, 1 / (1 + np.exp(-(lg - math.log(w)))), color=C_RED, ls="-.", lw=1.1, label=f"prior correction ($b$=$-$ln $w$={-math.log(w):.2f})")
    mp, ob, _ = reliability(wgt.y, wgt.raw)
    c.scatter(mp, ob, s=9, color=C_AMBER, zorder=3, label="observed (raw bins)")
    c.set_xlabel("raw score of weighted model"); c.set_ylabel("calibrated probability")
    c.set_xlim(0, 1); c.set_ylim(0, 1.3)  # headroom keeps the legend clear of the curve
    c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    c.legend(frameon=False, loc="upper left", borderaxespad=0.2)
    c.set_title("(c) Platt undoes the class weight", loc="left")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "fig_cs_cal.pdf"); plt.close(fig)
    return {"w": round(w, 4), "t_eq": round(t_eq, 4), "eq_none": {k: round(v, 4) for k, v in eq_none.items()},
            "eq_weight": {k: round(v, 4) for k, v in eq_w.items()}, "weighted_ece": out_ece,
            "platt_weighted": [round(sa, 4), round(sb, 4)], "ln_w": round(math.log(w), 4),
            "mean_pred": {"none_raw": round(float(none.raw.mean()), 4), "weight_raw": round(float(wgt.raw.mean()), 4),
                          "weight_platt": round(float(wgt.platt.mean()), 4)},
            "prevalence_2025": round(float(none.y.mean()), 4)}


def fig_threshold(out: Path) -> dict:
    """Recall, precision and expected cost of the final model (Platt) against the threshold."""
    cell = "F1full__none__lgbm"
    sel = load_cell(cell)
    ops = json.loads((CELLS / cell / "frozen_seed0.json").read_text())["thresholds"]["platt"]
    ts = np.round(np.arange(0.02, 0.801, 0.005), 3)
    cv = curve(sel.y, sel.platt, ts)
    fig, ax = plt.subplots(figsize=(COL_W - 0.06, 2.0))
    ax.plot(cv.t, cv.recall, color=C_GREEN, label="recall")
    ax.plot(cv.t, cv.precision, color=C_BLUE, label="precision")
    ax.plot(cv.t, cv.flagged, color=C_GREY, ls="--", lw=1, label="share flagged")
    ax.set_xlabel("threshold on calibrated probability"); ax.set_ylabel("rate"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 0.8)
    ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    ax2.plot(cv.t, cv.cost, color=C_RED, lw=1.2, label="expected cost (5:1)")
    ax2.set_ylabel("expected cost per collision", color=C_RED); ax2.tick_params(axis="y", colors=C_RED)
    ax2.set_ylim(0.5, 1.4)
    for key, lab, yy, ha, dx in [("tf2", "$t^{*}$", 0.04, "right", -0.006), ("tcost", "1/6", 0.04, "left", 0.006),
                                 ("t050", "0.5", 0.04, "left", 0.006)]:
        ax.axvline(ops[key], color="black", lw=0.6, ls=":")
        ax.text(ops[key] + dx, yy, lab, fontsize=8, ha=ha)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              handlelength=1.4, columnspacing=0.8)
    fig.savefig(out / "fig_threshold.pdf"); plt.close(fig)
    t_min = min_cost_threshold(sel.y, sel.platt, fn_cost=FN_COST, fp_cost=FP_COST)
    best = exact_point(sel.y, sel.platt, t_min)
    mp, ob, e = reliability(sel.y, sel.platt)
    raw_cost = {k: round(exact_point(sel.y, sel.raw, ops[k])["cost"], 4) for k in ("t050", "tcost")}
    return {"ops": {k: round(v, 4) for k, v in ops.items()}, "cost_min_t_2025": round(float(t_min), 4),
            "cost_min_2025": round(best["cost"], 4), "raw_cost_seed0": raw_cost, "final_ece_platt_seed0": round(e, 4),
            "final_mean_pred_platt": round(float(sel.platt.mean()), 4), "platt_final": list(map(lambda x: round(x, 4), sel.platt_coef))}


def fig_shap(out: Path, top: int = 12) -> dict:
    """Mean |SHAP| of the top features on CAL (2024) and TEST (2025)."""
    s = pd.read_csv(TABLES / "shap_importance.csv", index_col=0).sort_values("mean_abs_shap_test", ascending=False).head(top)
    labels = [SHAP_LABEL.get(i, i) for i in s.index][::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(COL_W - 0.83, 2.15))
    ax.barh(y + 0.19, s.mean_abs_shap_cal.to_numpy()[::-1], height=0.36, color=C_GREY, label="CAL (2024)")
    ax.barh(y - 0.19, s.mean_abs_shap_test.to_numpy()[::-1], height=0.36, color=C_BLUE, label="TEST (2025)")
    ax.set_yticks(y, labels); ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "fig_shap.pdf"); plt.close(fig)
    return {"top": list(s.index)}


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=_ROOT / "paper" / "figures")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    set_style()
    numbers = {"prauc": fig_prauc(args.out), "cs_cal": fig_cs_cal(args.out),
               "threshold": fig_threshold(args.out), "shap": fig_shap(args.out)}
    (args.out / "figure_numbers.json").write_text(json.dumps(numbers, indent=1))
    logger.info(json.dumps({k: v for k, v in numbers.items() if k != "prauc"}, indent=1))


if __name__ == "__main__":
    main()
