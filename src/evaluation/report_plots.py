"""Result figures for the report (style shared with the EDA figures via :mod:`src.evaluation.eda`).

Every function takes tidy result tables / prediction arrays and writes one PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src.evaluation.eda import PALETTE, add_note, add_titles, save_figure, set_style  # noqa: E402
from src.evaluation.metrics import reliability_curve  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_LABELS = {"lr": "Logistic Reg.", "dt": "Decision Tree", "rf": "Random Forest",
                "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost"}
MODEL_ORDER = ["lr", "dt", "rf", "xgb", "lgbm", "cat"]
CS_LABELS = {"none": "None", "weight": "Class weight", "rus": "Undersampling",
             "smotenc": "SMOTE-NC"}
VARIANT_LABELS = {"raw": "Uncalibrated", "platt": "Platt", "isotonic": "Isotonic"}


def pipeline_diagram(path: Path) -> Path:
    """Draw the end-to-end pipeline and temporal split design.

    Args:
        path: Output PNG.

    Returns:
        Written path.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    steps = ["STATS19\n2023–2025", "Clean +\nleakage audit", "Aggregate\nto collision",
             "Features\nF0 / F1", "6 models x\nimbalance (CS)", "Calibrate +\nthreshold (CAL)",
             "Freeze\nmanifest", "2025 test +\nstatistics"]
    for i, s in enumerate(steps):
        x = 0.15 + i * 1.35
        box = FancyBboxPatch((x, 2.75), 1.15, 1.0, boxstyle="round,pad=0.03,rounding_size=0.08",
                             fc=PALETTE.sequential[0] if i < 6 else PALETTE.sequential[2],
                             ec=PALETTE.axis)
        ax.add_patch(box)
        ax.text(x + 0.575, 3.25, s, ha="center", va="center", fontsize=8.6, color=PALETTE.ink)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.33, 3.25), xytext=(x + 1.17, 3.25),
                        arrowprops={"arrowstyle": "->", "color": PALETTE.ink_muted})
    splits = [("FIT\n2023 · 104k rows", 0.3, 3.6, PALETTE.series[0]),
              ("TUNE\n2024 odd weeks · 51k", 3.95, 2.0, PALETTE.series[2]),
              ("CAL\n2024 even weeks · 50k", 6.0, 2.0, PALETTE.ordinal[0]),
              ("TEST (sealed)\n2025 · 102k rows", 8.05, 2.75, PALETTE.series[1])]
    for label, x, w, c in splits:
        ax.add_patch(FancyBboxPatch((x, 1.2), w - 0.05, 0.75, boxstyle="round,pad=0.02",
                                    fc=c, ec="none", alpha=0.9))
        ax.text(x + (w - 0.05) / 2, 1.575, label, ha="center", va="center", fontsize=8.2,
                color="#ffffff", weight="bold", linespacing=1.3)
    ax.text(0.3, 0.75, "Encoders, selection, resampling: FIT (refit: FIT+TUNE).  HPO objective "
            "and early stopping: TUNE.  Calibrators and thresholds: CAL.", fontsize=8.6,
            color=PALETTE.ink_secondary)
    ax.text(0.3, 0.38, "TEST is read only after every decision is hashed into the frozen "
            "manifest.", fontsize=8.6, color=PALETTE.ink_secondary)
    add_titles(fig, "Pipeline and temporal evaluation design",
               "Train on the past, choose operating points on held-out 2024, score the unseen "
               "year 2025")
    fig.subplots_adjust(top=0.84)
    return save_figure(fig, path)


def forest_plot(df: pd.DataFrame, metric: str, path: Path, title: str, subtitle: str,
                xlabel: str, label_col: str = "label") -> Path:
    """Horizontal point-and-CI plot of paired deltas (A − B) from ``contrasts.csv``.

    Args:
        df: Rows with ``{metric}__delta``, ``__lo``, ``__hi``, ``p_holm`` and a label column.
        metric: Metric key prefix.
        path: Output PNG.
        title: Figure title.
        subtitle: Figure subtitle.
        xlabel: X-axis label.
        label_col: Column holding row labels.

    Returns:
        Written path.
    """
    df = df.reset_index(drop=True)
    if df.empty:
        logger.warning("No rows for %s; figure skipped", path.name)
        return path
    set_style()
    fig, ax = plt.subplots(figsize=(11.0, max(0.34 * len(df) + 1.9, 3.2)))
    y = np.arange(len(df))[::-1]
    d, lo, hi = df[f"{metric}__delta"], df[f"{metric}__lo"], df[f"{metric}__hi"]
    sig = (lo > 0) | (hi < 0)
    colors = [PALETTE.series[0] if s else PALETTE.ink_muted for s in sig]
    ax.hlines(y, lo, hi, color=colors, linewidth=2)
    ax.scatter(d, y, color=colors, zorder=3, s=28)
    ax.axvline(0, color=PALETTE.axis, linewidth=1)
    ax.set_yticks(y, df[label_col])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    for yi, di in zip(y, d, strict=True):
        ax.text(ax.get_xlim()[1], yi, f" {di:+.4f}", va="center", fontsize=8,
                color=PALETTE.ink_secondary)
    add_titles(fig, title, subtitle)
    add_note(fig, "Points: TEST (2025) paired difference A − B, seed 0; bars: 95% bootstrap CI "
                  "(B = 2000). Blue = CI excludes 0.")
    fig.subplots_adjust(top=1 - 0.95 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(),
                        left=0.42, right=0.90)
    return save_figure(fig, path)


def reliability_panel(y: np.ndarray, probs: dict[str, np.ndarray], path: Path, title: str,
                      subtitle: str) -> Path:
    """Reliability diagram (quantile bins) for several calibration variants.

    Args:
        y: Labels.
        probs: ``{variant: probabilities}``.
        path: Output PNG.
        title: Title.
        subtitle: Subtitle.

    Returns:
        Written path.
    """
    set_style()
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.6, 6.6), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], color=PALETTE.axis, linewidth=1, linestyle="--")
    for (v, p), c in zip(probs.items(), PALETTE.series, strict=False):
        rc = reliability_curve(y, p, n_bins=15, strategy="quantile")
        ax.plot(rc.mean_pred, rc.frac_pos, marker="o", ms=3.5, color=c,
                label=VARIANT_LABELS.get(v, v))
        axh.hist(p, bins=40, range=(0, 1), histtype="step", color=c, linewidth=1.3)
    ax.set_ylabel("Observed severe rate")
    ax.legend(loc="upper left")
    ax.grid()
    axh.set_xlabel("Predicted probability of a severe collision")
    axh.set_ylabel("Count")
    add_titles(fig, title, subtitle)
    fig.subplots_adjust(top=0.86, hspace=0.08)
    return save_figure(fig, path)


def pr_roc_curves(y: np.ndarray, probs: dict[str, np.ndarray], path: Path) -> Path:
    """PR and ROC curves for the best configuration of each model family on TEST.

    Args:
        y: Labels.
        probs: ``{label: probabilities}``.
        path: Output PNG.

    Returns:
        Written path.
    """
    set_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = list(PALETTE.series) + list(PALETTE.ordinal)
    for (lab, p), c in zip(probs.items(), colors, strict=False):
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        keep = rc >= 0.01  # the first few top-scored collisions make precision jump around
        a1.plot(rc[keep], pr[keep], color=c, linewidth=1.4, label=lab)
        a2.plot(fpr, tpr, color=c, linewidth=1.4, label=lab)
    a1.axhline(y.mean(), color=PALETTE.axis, linestyle="--", linewidth=1)
    a1.text(0.99, y.mean() + 0.01, f"prevalence {y.mean():.3f}", ha="right", fontsize=8,
            color=PALETTE.ink_muted)
    a2.plot([0, 1], [0, 1], color=PALETTE.axis, linestyle="--", linewidth=1)
    a1.set(xlabel="Recall (severe)", ylabel="Precision", title="Precision–recall")
    a2.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    a1.legend(fontsize=8)
    for a in (a1, a2):
        a.grid()
    add_titles(fig, "Discrimination on the unseen 2025 test year",
               "Best CAL configuration per model family (F1 features, CAL-selected CS)")
    fig.subplots_adjust(top=0.8, wspace=0.25)
    return save_figure(fig, path)


def threshold_curves(y: np.ndarray, p: np.ndarray, thresholds: dict[str, float],
                     path: Path, title: str, subtitle: str) -> Path:
    """Precision / recall / F2 as a function of the decision threshold with frozen cut-offs.

    Args:
        y: Labels.
        p: Probabilities.
        thresholds: ``{name: threshold}`` markers.
        path: Output PNG.
        title: Title.
        subtitle: Subtitle.

    Returns:
        Written path.
    """
    set_style()
    grid = np.linspace(0.01, 0.95, 189)
    pred = p[None, :] >= grid[:, None]
    tp = (pred & (y == 1)).sum(1)
    fp = (pred & (y == 0)).sum(1)
    fn = ((~pred) & (y == 1)).sum(1)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)
    rec = tp / np.maximum(tp + fn, 1)
    f2 = np.where(prec + rec > 0, 5 * prec * rec / np.maximum(4 * prec + rec, 1e-12), 0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for arr, lab, c in ((prec, "Precision", PALETTE.series[0]), (rec, "Recall", PALETTE.series[1]),
                        (f2, "F2", PALETTE.series[2])):
        ax.plot(grid, arr, color=c, label=lab, linewidth=1.6)
    for name, t in thresholds.items():
        ax.axvline(t, color=PALETTE.ink_muted, linestyle=":", linewidth=1)
        ax.text(t, 1.02, name, rotation=90, fontsize=7.5, color=PALETTE.ink_secondary,
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.set(xlabel="Decision threshold", ylabel="Metric value", ylim=(0, 1))
    ax.legend(loc="center right")
    ax.grid()
    add_titles(fig, title, subtitle)
    fig.subplots_adjust(top=0.78)
    return save_figure(fig, path)


def confusion_pair(y: np.ndarray, p: np.ndarray, cuts: dict[str, float], path: Path) -> Path:
    """Side-by-side confusion matrices at two operating thresholds.

    Args:
        y: Labels.
        p: Probabilities.
        cuts: ``{label: threshold}`` (two entries).
        path: Output PNG.

    Returns:
        Written path.
    """
    set_style()
    fig, axes = plt.subplots(1, len(cuts), figsize=(4.4 * len(cuts), 4.2))
    for ax, (lab, t) in zip(np.atleast_1d(axes), cuts.items(), strict=True):
        pr = (p >= t).astype(int)
        cm = np.array([[np.sum((y == 0) & (pr == 0)), np.sum((y == 0) & (pr == 1))],
                       [np.sum((y == 1) & (pr == 0)), np.sum((y == 1) & (pr == 1))]])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]:,}\n({share:.1%})", ha="center", va="center",
                        fontsize=9, color="#ffffff" if share > 0.55 else PALETTE.ink)
        ax.set_xticks([0, 1], ["Pred. slight", "Pred. severe"])
        ax.set_yticks([0, 1], ["Slight", "Severe"])
        ax.set_title(f"{lab} (t = {t:.3f})", fontsize=10)
    add_titles(fig, "Confusion matrices on 2025 for the selected model",
               "Row-normalised shares in brackets")
    fig.subplots_adjust(top=0.78, wspace=0.35)
    return save_figure(fig, path)


def val_test_scatter(gap: pd.DataFrame, metric: str, path: Path) -> Path:
    """CAL vs TEST metric per trained cell (raw scores), coloured by model family.

    Args:
        gap: ``val_test_gap.csv`` rows (variant raw, op t050).
        metric: ``pr_auc`` or ``roc_auc``.
        path: Output PNG.

    Returns:
        Written path.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    colors = dict(zip(MODEL_ORDER, list(PALETTE.series) + list(PALETTE.ordinal), strict=False))
    for m, g in gap.groupby("model"):
        ax.scatter(g[f"{metric}_cal"], g[f"{metric}_test"], s=26, color=colors.get(m),
                   label=MODEL_LABELS.get(m, m), alpha=0.9)
    lo = min(gap[f"{metric}_cal"].min(), gap[f"{metric}_test"].min()) - 0.01
    hi = max(gap[f"{metric}_cal"].max(), gap[f"{metric}_test"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=PALETTE.axis, linestyle="--", linewidth=1)
    ax.set(xlabel=f"CAL 2024 {metric.upper().replace('_', '-')}",
           ylabel=f"TEST 2025 {metric.upper().replace('_', '-')}", xlim=(lo, hi), ylim=(lo, hi))
    ax.legend(fontsize=8)
    ax.grid()
    add_titles(fig, "Does validation performance carry over to 2025?",
               "One point per trained cell (mean of 3 seeds); dashed = no change")
    fig.subplots_adjust(top=0.86)
    return save_figure(fig, path)


def ablation_heatmap(table: pd.DataFrame, metric: str, path: Path, title: str) -> Path:
    """Heatmap of a TEST metric: model family x (feature set, CS).

    Args:
        table: Rows with ``model, feature_set, cs`` and ``metric`` (mean over seeds).
        metric: Column to show.
        path: Output PNG.
        title: Title.

    Returns:
        Written path.
    """
    set_style()
    t = table.copy()
    t["config"] = t["feature_set"] + " · " + t["cs"].map(CS_LABELS).fillna(t["cs"])
    pv = t.pivot_table(index="model", columns="config", values=metric).reindex(
        [m for m in MODEL_ORDER if m in set(t["model"])])
    order = [f"{fs} · {CS_LABELS[cs]}" for fs in ("F0", "F1") for cs in CS_LABELS]
    pv = pv[[c for c in order if c in pv.columns]]
    fig, ax = plt.subplots(figsize=(1.25 * pv.shape[1] + 2.4, 0.55 * pv.shape[0] + 1.9))
    im = ax.imshow(pv.to_numpy(), cmap="Blues", aspect="auto")
    vals = pv.to_numpy()
    mid = np.nanmean(vals)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if vals[i, j] > mid else PALETTE.ink)
    ax.set_xticks(range(pv.shape[1]), pv.columns, rotation=35, ha="right")
    ax.set_yticks(range(pv.shape[0]), [MODEL_LABELS.get(m, m) for m in pv.index])
    fig.colorbar(im, ax=ax, fraction=0.03)
    add_titles(fig, title, "TEST 2025, mean of 3 seeds")
    fig.subplots_adjust(top=1 - 0.8 / fig.get_figheight(), bottom=0.3, left=0.16)
    return save_figure(fig, path)


def stratified_bars(strat: pd.DataFrame, path: Path) -> Path:
    """Recall / precision of the selected model by reporting system and police attendance.

    Args:
        strat: ``test_stratified.csv``.
        path: Output PNG.

    Returns:
        Written path.
    """
    set_style()
    lab = {("collision_injury_based", 1): "Injury-based recording",
           ("collision_injury_based", 0): "Officer-judged recording",
           ("police_attended", 1): "Officer attended",
           ("police_attended", 2): "Not attended",
           ("police_attended", 3): "Self-reported"}
    s = strat.copy()
    s["label"] = [lab.get((a, int(b)), f"{a}={b}") for a, b in zip(s["stratifier"], s["level"],
                                                                     strict=True)]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(s))
    for k, (m, c) in enumerate((("recall", PALETTE.series[0]), ("precision", PALETTE.series[1]),
                                ("prevalence", PALETTE.ink_muted))):
        ax.bar(x + (k - 1) * 0.26, s[m], width=0.25, color=c, label=m.capitalize())
    ax.set_xticks(x, [f"{a}\n(n={n:,})" for a, n in zip(s["label"], s["n"], strict=True)],
                  fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y")
    add_titles(fig, "Selected model on 2025 by recording regime and attendance",
               "Same frozen threshold everywhere; prevalence differs strongly between strata")
    fig.subplots_adjust(top=0.82)
    return save_figure(fig, path)
