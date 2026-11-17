"""Reusable exploratory-data-analysis helpers for the collision-severity project.

Protocol (plan section 2): 2025 is the sealed temporal test year. Every function that touches the
label (``target`` / ``severity3``) goes through :func:`dev_rows`, which keeps 2023-2024 only and
raises if a 2025 row slips through. 2025 appears only in label-free covariate views
(reporting-system share, missingness).

Plot styling follows one fixed palette (:data:`PALETTE`), validated with the dataviz skill's
palette checker: categorical blue / orange / aqua, an ordinal blue ramp for Fatal / Serious /
Slight,
a one-hue blue sequential ramp and a blue <-> red diverging map with a neutral grey midpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm

from src.data.codebook import Codebook

logger = logging.getLogger(__name__)

DEV_YEARS: tuple[int, ...] = (2023, 2024)
TEST_YEAR: int = 2025
ALL_YEARS: tuple[int, ...] = (*DEV_YEARS, TEST_YEAR)
MISSING_LABEL = "Missing / unknown"
ArrayLike1D = float | int | Sequence[float] | np.ndarray | pd.Series
DPI = 200

# Vehicle-type codes (data guide, vehicle.vehicle_type) used for vehicle-mix flags.
MOTORCYCLE_CODES: tuple[int, ...] = (2, 3, 4, 5, 23, 97)
PEDAL_CYCLE_CODES: tuple[int, ...] = (1,)
HGV_CODES: tuple[int, ...] = (20, 21, 98)

# Readable short forms for long data-guide labels (keys are the exact guide labels).
SHORT_LABELS: dict[str, str] = {
    "Not at junction or within 20 metres": "Not at junction",
    "Junction with more than four arms (not roundabout)": "Junction >4 arms",
    "Using private drive or entrance": "Private drive / entrance",
    "T or staggered junction": "T / staggered junction",
    "Flood over 3cm. deep": "Flood >3 cm",
    "Fine no high winds": "Fine",
    "Raining no high winds": "Raining",
    "Snowing no high winds": "Snowing",
    "Fine + high winds": "Fine + high winds",
    "Raining + high winds": "Raining + high winds",
    "Snowing + high winds": "Snowing + high winds",
    "Darkness - lights lit": "Dark, lights lit",
    "Darkness - lights unlit": "Dark, lights unlit",
    "Darkness - no lighting": "Dark, no lighting",
    "Darkness - lighting unknown": "Dark, lighting unknown",
    "Based on severity reporting": "Severity-based (legacy)",
    "Based on Injury code reporting": "Injury-based (CRaSH / COPA)",
    "No - accident was reported using a self completion  form (self rep only)": (
        "No, self-completion report"
    ),
}


@dataclass(frozen=True)
class Palette:
    """Validated chart colours (light surface; dataviz reference palette).

    Attributes:
        surface: Chart / figure background.
        ink: Primary text.
        ink_secondary: Secondary text (subtitles, n labels).
        ink_muted: Axis ticks, reference annotations.
        grid: Hairline gridlines.
        axis: Baseline / axis rules.
        series: Categorical slots in fixed order (blue, orange, aqua).
        ordinal: Ordinal blue ramp, dark -> light (Fatal, Serious, Slight).
        sequential: Blue ramp steps 100 -> 700 for magnitude.
        diverging: (low pole, neutral midpoint, high pole).
        masked: Fill for cells suppressed by a minimum-count rule.
    """

    surface: str = "#fcfcfb"
    ink: str = "#0b0b0b"
    ink_secondary: str = "#52514e"
    ink_muted: str = "#898781"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"
    series: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
    ordinal: tuple[str, ...] = ("#0d366b", "#2a78d6", "#86b6ef")
    sequential: tuple[str, ...] = (
        "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
    )
    diverging: tuple[str, str, str] = ("#2a78d6", "#f0efec", "#e34948")
    masked: str = "#e1e0d9"


PALETTE = Palette()


@dataclass
class SummaryCollector:
    """Accumulates key rates (with CIs and n) for ``eda_summary.csv``.

    Attributes:
        rows: One dict per reported rate.
    """

    rows: list[dict[str, object]] = field(default_factory=list)

    def add(
        self, table: pd.DataFrame, section: str, variable: str, years: str, figure: str
    ) -> None:
        """Append a rate table produced by :func:`rate_table`.

        Args:
            table: Output of :func:`rate_table` (columns level, n, n_pos, rate, ci_low, ci_high).
            section: Short analysis name (e.g. ``"road"``).
            variable: Variable the rates are grouped by.
            years: Years covered, e.g. ``"2023-2024"``.
            figure: Figure file the rates appear in.
        """
        for row in table.itertuples(index=False):
            self.rows.append(
                {
                    "section": section,
                    "variable": variable,
                    "level": row.level,
                    "years": years,
                    "n": int(row.n),
                    "n_pos": int(row.n_pos),
                    "rate": round(float(row.rate), 5),
                    "ci_low": round(float(row.ci_low), 5),
                    "ci_high": round(float(row.ci_high), 5),
                    "figure": figure,
                }
            )

    def to_frame(self) -> pd.DataFrame:
        """Return the collected rows as a DataFrame."""
        return pd.DataFrame(self.rows)


# --------------------------------------------------------------------------------------------
# Statistics and data shaping
# --------------------------------------------------------------------------------------------
def wilson_ci(
    k: ArrayLike1D, n: ArrayLike1D, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial proportions.

    Args:
        k: Number of successes (scalar or array).
        n: Number of trials (scalar or array, > 0).
        confidence: Two-sided confidence level.

    Returns:
        Tuple ``(lower, upper)`` arrays.
    """
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    if np.any(n_arr <= 0):
        raise ValueError("wilson_ci requires n > 0")
    z = norm.ppf(0.5 + confidence / 2)
    p = k_arr / n_arr
    denom = 1 + z**2 / n_arr
    centre = (p + z**2 / (2 * n_arr)) / denom
    half = z * np.sqrt(p * (1 - p) / n_arr + z**2 / (4 * n_arr**2)) / denom
    return centre - half, centre + half


def dev_rows(df: pd.DataFrame, years: Iterable[int] = DEV_YEARS) -> pd.DataFrame:
    """Return development-year rows only, refusing the sealed test year.

    Args:
        df: Collision-level frame with ``collision_year``.
        years: Years to keep; must not contain the test year.

    Returns:
        Filtered copy.

    Raises:
        ValueError: If ``years`` contains the sealed test year.
    """
    years = tuple(years)
    if TEST_YEAR in years:
        raise ValueError(f"Label-based EDA must not use the sealed test year {TEST_YEAR}")
    out = df.loc[df["collision_year"].isin(years)].copy()
    assert not (out["collision_year"] == TEST_YEAR).any()
    return out


def rate_table(
    df: pd.DataFrame,
    by: str | list[str],
    target: str = "target",
    min_n: int = 1,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Positive rate of ``target`` per group, with Wilson CIs.

    Missing group values form their own level (``MISSING_LABEL`` after labelling).

    Args:
        df: Input frame.
        by: Grouping column name(s).
        target: Binary 0/1 column.
        min_n: Groups with fewer rows are dropped.
        confidence: CI level.

    Returns:
        DataFrame with the group column(s) plus ``n, n_pos, rate, ci_low, ci_high``.
    """
    keys = [by] if isinstance(by, str) else list(by)
    grouped = df.groupby(keys, dropna=False, observed=True)[target].agg(n="size", n_pos="sum")
    grouped = grouped.reset_index()
    grouped = grouped.loc[grouped["n"] >= min_n].copy()
    grouped["rate"] = grouped["n_pos"] / grouped["n"]
    low, high = wilson_ci(grouped["n_pos"].to_numpy(), grouped["n"].to_numpy(), confidence)
    grouped["ci_low"] = low
    grouped["ci_high"] = high
    return grouped.reset_index(drop=True)


def short_label(label: str) -> str:
    """Return a compact, readable version of a data-guide label."""
    return SHORT_LABELS.get(label, label)


def code_labeller(codebook: Codebook, table: str, name: str) -> Mapping[object, str]:
    """Return ``{code: short label}`` for one field (codes as ints).

    Args:
        codebook: Parsed data guide.
        table: Guide table (``"collision"``, ``"vehicle"``, ``"casualty"``).
        name: Field name.

    Returns:
        Mapping from code to readable label.
    """
    return {code: short_label(lab) for code, lab in codebook.field_labels(table, name).items()}


def label_levels(values: pd.Series, labels: Mapping[object, str], fmt: str = "{}") -> pd.Series:
    """Map codes to labels; missing values become :data:`MISSING_LABEL`.

    Args:
        values: Code series (nullable ints allowed).
        labels: ``{code: label}``; unmapped codes fall back to ``fmt.format(code)``.
        fmt: Fallback format for codes without a label (e.g. ``"{} mph"``).

    Returns:
        Object series of labels.
    """

    def _one(value: object) -> str:
        if pd.isna(value):
            return MISSING_LABEL
        code = int(value) if isinstance(value, (int, np.integer, float)) else value
        return labels.get(code, fmt.format(code))

    return values.map(_one).astype(object)


def vehicle_mix_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Derive collision-level vehicle-mix indicators from ``veh_vehicle_type__<code>`` counts.

    Args:
        df: Collision frame with vehicle-type count columns and ``number_of_vehicles``.

    Returns:
        Frame with string columns ``pedestrian, motorcycle, pedal_cycle, hgv, vehicles``.
    """

    def _any(codes: Iterable[int]) -> pd.Series:
        cols = [f"veh_vehicle_type__{c}" for c in codes if f"veh_vehicle_type__{c}" in df]
        return df[cols].fillna(0).sum(axis=1) > 0

    yes_no = {True: "Involved", False: "Not involved"}
    n_veh = df["number_of_vehicles"].astype(float)
    return pd.DataFrame(
        {
            "pedestrian": (df["ped_involved"] == 1).map(yes_no),
            "motorcycle": _any(MOTORCYCLE_CODES).map(yes_no),
            "pedal_cycle": _any(PEDAL_CYCLE_CODES).map(yes_no),
            "hgv": _any(HGV_CODES).map(yes_no),
            "vehicles": np.select(
                [n_veh <= 1, n_veh == 2], ["Single vehicle", "Two vehicles"], "Three or more"
            ),
        },
        index=df.index,
    )


def missing_share(
    frame: pd.DataFrame, fields: Sequence[str], missing_codes: Mapping[str, frozenset[int]]
) -> pd.Series:
    """Share of rows whose value is NaN or a field-specific missing code.

    Args:
        frame: Raw table slice.
        fields: Columns to evaluate (absent columns count as 100% missing).
        missing_codes: ``{field: codes}`` treated as missing (``-1`` should be included).

    Returns:
        Series indexed by field with the missing share in [0, 1].
    """
    shares: dict[str, float] = {}
    for name in fields:
        if name not in frame:
            shares[name] = 1.0
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        codes = list(missing_codes.get(name, frozenset({-1})))
        shares[name] = float((values.isna() | values.isin(codes)).mean())
    return pd.Series(shares)


# --------------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------------
def set_style(palette: Palette = PALETTE) -> None:
    """Apply the shared, low-chrome matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": palette.surface,
            "axes.facecolor": palette.surface,
            "savefig.facecolor": palette.surface,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": palette.ink,
            "axes.labelcolor": palette.ink_secondary,
            "axes.edgecolor": palette.axis,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": palette.ink,
            "axes.grid": False,
            "grid.color": palette.grid,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": palette.ink_muted,
            "ytick.color": palette.ink_muted,
            "xtick.labelcolor": palette.ink_secondary,
            "ytick.labelcolor": palette.ink_secondary,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def add_titles(fig: Figure, title: str, subtitle: str, palette: Palette = PALETTE) -> None:
    """Add a left-aligned figure title and a secondary-ink subtitle."""
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / height, title, ha="left", va="top", fontsize=13, weight="bold",
             color=palette.ink)
    fig.text(0.01, 1 - 0.42 / height, subtitle, ha="left", va="top", fontsize=9.5,
             color=palette.ink_secondary)


def add_note(fig: Figure, note: str, palette: Palette = PALETTE) -> None:
    """Add a muted footnote at the bottom-left of a figure."""
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8, color=palette.ink_muted)


def save_figure(fig: Figure, path: Path) -> Path:
    """Save a figure at :data:`DPI` and close it.

    Args:
        fig: Figure to save.
        path: Output PNG path (parents created).

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    logger.info(f"Wrote figure {path}")
    return path


def pct_axis(ax: Axes, axis: str = "x", decimals: int = 0) -> None:
    """Format an axis as percentages of a 0-1 quantity."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def dot_whisker(
    ax: Axes,
    table: pd.DataFrame,
    label_col: str = "level",
    reference: float | None = None,
    color: str | Sequence[str] = PALETTE.series[0],
    show_n: bool = True,
    palette: Palette = PALETTE,
) -> None:
    """Horizontal dot-and-whisker plot of rates with CIs (first row on top).

    Args:
        ax: Target axes.
        table: Rate table with ``label_col, n, rate, ci_low, ci_high`` in display order.
        label_col: Column holding category labels.
        reference: Optional overall rate drawn as a solid hairline.
        color: One colour, or one per row.
        show_n: Append ``n=`` to each tick label.
        palette: Colours for chrome.
    """
    y = np.arange(len(table))[::-1]
    colors = [color] * len(table) if isinstance(color, str) else list(color)
    if reference is not None:
        ax.axvline(reference, color=palette.ink_muted, linewidth=1.0, zorder=1)
    ax.hlines(y, table["ci_low"], table["ci_high"], colors=colors, linewidth=2, zorder=2)
    ax.scatter(
        table["rate"], y, s=42, c=colors, zorder=3, edgecolors=palette.surface, linewidths=1.5
    )
    if show_n:
        ticks = [f"{lab}  (n={n:,})" for lab, n in zip(table[label_col], table["n"], strict=True)]
    else:
        ticks = list(table[label_col])
    ax.set_yticks(y, ticks)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(table) - 0.4)
    pct_axis(ax, "x")


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
def plot_class_balance(dev: pd.DataFrame, path: Path, collector: SummaryCollector) -> Path:
    """Figure 1: Fatal / Serious / Slight composition per development year.

    Args:
        dev: 2023-2024 collision rows.
        path: Output PNG.
        collector: Summary sink.

    Returns:
        Written path.
    """
    dev = dev_rows(dev)
    names = {1: "Fatal", 2: "Serious", 3: "Slight"}
    counts = dev.groupby(["collision_year", "severity3"]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    severe = rate_table(dev, "collision_year").rename(columns={"collision_year": "level"})
    collector.add(severe, "class_balance", "collision_year", "2023-2024", path.name)
    comp = dev.assign(level=dev["severity3"].map(names))
    for year, part in comp.groupby("collision_year"):
        for sev in (1, 2):
            tab = rate_table(
                part.assign(level="all", t=(part["severity3"] == sev).astype(int)), "level", "t"
            )
            tab["level"] = names[sev]
            collector.add(tab, "class_balance", "severity3_share", str(year), path.name)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.subplots_adjust(left=0.1, right=0.78, top=0.62, bottom=0.12)
    years = list(shares.index)
    y = np.arange(len(years))[::-1]
    left = np.zeros(len(years))
    for sev, colour in zip((1, 2, 3), PALETTE.ordinal, strict=True):
        width = shares[sev].to_numpy()
        ax.barh(y, width, left=left, height=0.55, color=colour, edgecolor=PALETTE.surface,
                linewidth=2, label=names[sev])
        for yi, lo, wi, n in zip(y, left, width, counts[sev].to_numpy(), strict=True):
            text = f"{names[sev]} {wi:.1%}\n(n={n:,})"
            if wi > 0.12:
                ink = "#ffffff" if sev < 3 else PALETTE.ink
                ax.text(lo + wi / 2, yi, text, ha="center", va="center", fontsize=9, color=ink)
            else:
                ax.annotate(text, (lo + wi / 2, yi + 0.28), xytext=(0, 6),
                            textcoords="offset points", ha="left", va="bottom", fontsize=8.5,
                            color=PALETTE.ink_secondary)
        left += width
    for yi, row in zip(y, severe.itertuples(index=False), strict=True):
        ax.text(1.02, yi, f"Severe {row.rate:.1%}\n[{row.ci_low:.1%}, {row.ci_high:.1%}]\n"
                f"n = {row.n:,}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9, color=PALETTE.ink)
    ax.set_yticks(y, [str(yr) for yr in years])
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.set_xlim(0, 1)
    pct_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.12), handlelength=1.2)
    add_titles(fig, "About one collision in four is severe",
               "Collision severity composition, STATS19 2023–2024 (development years). "
               "Severe = Fatal + Serious.")
    return save_figure(fig, path)


def plot_monthly_trend(
    df: pd.DataFrame, path: Path, collector: SummaryCollector
) -> Path:
    """Figure 2: monthly severe rate (2023-24) above injury-based reporting share (2023-25).

    Two stacked panels on a shared time axis instead of a dual-axis overlay; the reporting share
    is label-free and therefore includes the sealed test year.

    Args:
        df: All collision rows (2023-2025).
        path: Output PNG.
        collector: Summary sink.

    Returns:
        Written path.
    """
    period = df["date"].dt.to_period("M").dt.to_timestamp()
    dev = dev_rows(df.assign(period=period))
    rate = rate_table(dev, "period")
    share = rate_table(df.assign(period=period), "period", target="collision_injury_based")
    yearly_share = rate_table(df, "collision_year", target="collision_injury_based")
    collector.add(yearly_share.rename(columns={"collision_year": "level"}),
                  "reporting_drift", "injury_based_share", "2023-2025", path.name)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.07, hspace=0.32)
    test_start = pd.Timestamp(f"{TEST_YEAR}-01-01")
    test_end = df["date"].max() + pd.Timedelta(days=15)
    for ax in (ax1, ax2):
        ax.axvspan(test_start - pd.Timedelta(days=15), test_end, color=PALETTE.grid, alpha=0.45,
                   linewidth=0, zorder=0)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    blue, orange = PALETTE.series[0], PALETTE.series[1]
    ax1.fill_between(rate["period"], rate["ci_low"], rate["ci_high"], color=blue, alpha=0.18,
                     linewidth=0)
    ax1.plot(rate["period"], rate["rate"], color=blue, linewidth=2, marker="o", markersize=4)
    overall = dev["target"].mean()
    ax1.axhline(overall, color=PALETTE.ink_muted, linewidth=1)
    ax1.text(test_start, overall + 0.003, f"  2023–24 mean {overall:.1%}",
             fontsize=8.5, color=PALETTE.ink_secondary, va="bottom", ha="left")
    ax1.text(test_start + pd.Timedelta(days=170), 0.8, "2025 sealed test year\n"
             "(severity not inspected)", transform=ax1.get_xaxis_transform(), ha="center",
             va="center", fontsize=9, color=PALETTE.ink_secondary)
    ax1.set_title("Severe rate (Fatal + Serious) per month, 95% Wilson CI band", pad=6)
    pct_axis(ax1, "y")
    lo = float(rate["ci_low"].min())
    hi = float(rate["ci_high"].max())
    ax1.set_ylim(np.floor(lo * 50) / 50, np.ceil(hi * 50) / 50)

    ax2.plot(share["period"], share["rate"], color=orange, linewidth=2, marker="o", markersize=4)
    for row in yearly_share.itertuples(index=False):
        mid = pd.Timestamp(f"{row.collision_year}-07-01")
        ax2.text(mid, 0.96, f"{row.collision_year}: {row.rate:.0%}", ha="center", va="top",
                 fontsize=9, color=PALETTE.ink, transform=ax2.get_xaxis_transform())
    ax2.set_title("Share of collisions recorded by injury-based systems (CRaSH / COPA) — "
                  "label-free, incl. 2025", pad=6)
    ax2.set_ylim(0, 1.08)
    pct_axis(ax2, "y")
    add_titles(fig, "Severe rate edged up in 2024 while injury-based recording spread",
               "STATS19 collisions by month. Severity shown for 2023–2024 only; the reporting-"
               "system share is a covariate and is shown through 2025.")
    return save_figure(fig, path)


def plot_hour_dow_heatmap(
    dev: pd.DataFrame, path: Path, collector: SummaryCollector, codebook: Codebook,
    min_n: int = 100,
) -> Path:
    """Figure 3: hour-of-day x day-of-week severe-rate heatmap with a count threshold.

    Args:
        dev: 2023-2024 rows.
        path: Output PNG.
        collector: Summary sink.
        codebook: Data guide (day labels).
        min_n: Cells with fewer collisions are masked.

    Returns:
        Written path.
    """
    dev = dev_rows(dev)
    table = rate_table(dev, ["day_of_week", "hour"])
    days = [2, 3, 4, 5, 6, 7, 1]  # Monday first
    day_names = code_labeller(codebook, "collision", "day_of_week")
    rate = table.pivot(index="day_of_week", columns="hour", values="rate").reindex(days)
    count = table.pivot(index="day_of_week", columns="hour", values="n").reindex(days)
    masked = count < min_n
    by_hour = rate_table(dev, "hour").rename(columns={"hour": "level"})
    collector.add(by_hour, "temporal", "hour", "2023-2024", path.name)
    by_day = rate_table(dev, "day_of_week")
    by_day["level"] = by_day["day_of_week"].map(day_names)
    collector.add(by_day.drop(columns="day_of_week"), "temporal", "day_of_week", "2023-2024",
                  path.name)

    cmap = LinearSegmentedColormap.from_list("seq_blue", PALETTE.sequential)
    cmap.set_bad(PALETTE.masked)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.82, bottom=0.14)
    data = np.ma.masked_array(rate.to_numpy(dtype=float), mask=masked.to_numpy())
    mesh = ax.pcolormesh(np.arange(25), np.arange(8), data[::-1], cmap=cmap,
                         edgecolors=PALETTE.surface, linewidth=1.5)
    ax.set_xticks(np.arange(24) + 0.5, [f"{h:02d}" for h in range(24)])
    ax.set_yticks(np.arange(7) + 0.5, [day_names[d] for d in days][::-1])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Hour of day (collision time)")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    cbar.outline.set_visible(False)
    cbar.set_label("Severe rate", color=PALETTE.ink_secondary)
    peak = table.loc[table["n"] >= min_n].sort_values("rate").iloc[-1]
    low = table.loc[table["n"] >= min_n].sort_values("rate").iloc[0]
    for row, tag in ((peak, "highest"), (low, "lowest")):
        xi = int(row["hour"]) + 0.5
        yi = 6 - days.index(int(row["day_of_week"])) + 0.5
        ink = "#ffffff" if tag == "highest" else PALETTE.ink
        ax.text(xi, yi, f"{row['rate']:.0%}", ha="center", va="center", fontsize=8, color=ink,
                weight="bold")
    n_masked = int(masked.to_numpy().sum())
    mask_note = (f"{n_masked} grey cells have < {min_n} collisions" if n_masked
                 else f"all cells have ≥ {min_n} collisions")
    add_titles(fig, "Severity peaks at night and on Sundays; weekday commute hours are lowest",
               f"Severe rate by hour and day of week, 2023–2024 (n = {len(dev):,}); {mask_note}. "
               "Labelled cells: highest and lowest rate.")
    return save_figure(fig, path)


@dataclass(frozen=True)
class FactorSpec:
    """One categorical factor for a small-multiples rate panel.

    Attributes:
        column: Column in the collision frame.
        title: Panel title.
        ordered: Keep code order (ordinal factor) instead of sorting by rate.
        fallback_fmt: Format for codes without a guide label.
        min_n: Levels with fewer rows are dropped from the plot.
    """

    column: str
    title: str
    ordered: bool = False
    fallback_fmt: str = "{}"
    min_n: int = 200


def factor_table(dev: pd.DataFrame, spec: FactorSpec, codebook: Codebook) -> pd.DataFrame:
    """Labelled rate table for one factor, in display order.

    Args:
        dev: 2023-2024 rows.
        spec: Factor description.
        codebook: Data guide.

    Returns:
        Rate table with a ``level`` label column.
    """
    labels = code_labeller(codebook, "collision", spec.column)
    table = rate_table(dev, spec.column, min_n=spec.min_n)
    table["level"] = label_levels(table[spec.column], labels, spec.fallback_fmt)
    is_missing = table["level"] == MISSING_LABEL
    if spec.ordered:
        body = table.loc[~is_missing].sort_values(spec.column)
    else:
        body = table.loc[~is_missing].sort_values("rate", ascending=False)
    return pd.concat([body, table.loc[is_missing]], ignore_index=True)


def plot_factor_panels(
    dev: pd.DataFrame,
    specs: Sequence[FactorSpec],
    path: Path,
    collector: SummaryCollector,
    codebook: Codebook,
    title: str,
    section: str,
    ncols: int = 2,
) -> Path:
    """Figures 4-5: small multiples of severe rate by categorical factor.

    Args:
        dev: 2023-2024 rows.
        specs: Factors to show (one panel each).
        path: Output PNG.
        collector: Summary sink.
        codebook: Data guide.
        title: Figure headline.
        section: Summary section name.
        ncols: Panel columns.

    Returns:
        Written path.
    """
    dev = dev_rows(dev)
    overall = float(dev["target"].mean())
    tables = [factor_table(dev, spec, codebook) for spec in specs]
    for spec, table in zip(specs, tables, strict=True):
        collector.add(table, section, spec.column, "2023-2024", path.name)
    nrows = int(np.ceil(len(specs) / ncols))
    heights = [max(len(t) for t in tables[r * ncols:(r + 1) * ncols]) + 1.5 for r in range(nrows)]
    fig = plt.figure(figsize=(12, 0.32 * sum(heights) + 1.6))
    grid = fig.add_gridspec(nrows, ncols, height_ratios=heights, hspace=0.55, wspace=0.75,
                            left=0.2, right=0.97, top=1 - 0.95 / fig.get_figheight(),
                            bottom=0.55 / fig.get_figheight())
    xmax = max(float(t["ci_high"].max()) for t in tables)
    xmax = min(1.0, np.ceil((xmax + 0.02) * 20) / 20)
    for i, (spec, table) in enumerate(zip(specs, tables, strict=True)):
        ax = fig.add_subplot(grid[i // ncols, i % ncols])
        dot_whisker(ax, table, reference=overall)
        ax.set_xlim(0, xmax)
        ax.set_title(spec.title, pad=4)
    add_titles(fig, title,
               f"Severe rate with 95% Wilson CI, 2023–2024 (n = {len(dev):,}). Vertical line = "
               f"overall rate {overall:.1%}. Levels with < {specs[0].min_n} collisions omitted.")
    return save_figure(fig, path)


def plot_vehicle_mix(dev: pd.DataFrame, path: Path, collector: SummaryCollector) -> Path:
    """Figure 6: severe rate by vulnerable-road-user involvement and vehicle count.

    Args:
        dev: 2023-2024 rows.
        path: Output PNG.
        collector: Summary sink.

    Returns:
        Written path.
    """
    dev = dev_rows(dev)
    flags = vehicle_mix_flags(dev)
    frame = pd.concat([dev[["target"]], flags], axis=1)
    overall = float(dev["target"].mean())
    groups = [
        ("pedestrian", "Pedestrian casualty", ["Involved", "Not involved"]),
        ("motorcycle", "Motorcycle (incl. electric)", ["Involved", "Not involved"]),
        ("pedal_cycle", "Pedal cycle", ["Involved", "Not involved"]),
        ("hgv", "Goods vehicle > 3.5 t", ["Involved", "Not involved"]),
        ("vehicles", "Vehicles in collision", ["Single vehicle", "Two vehicles", "Three or more"]),
    ]
    fig = plt.figure(figsize=(9, 7.4))
    heights = [len(order) + 1.2 for *_, order in groups]
    grid = fig.add_gridspec(len(groups), 1, height_ratios=heights, hspace=0.55, left=0.3,
                            right=0.95, top=0.88, bottom=0.05)
    for i, (col, title, order) in enumerate(groups):
        table = rate_table(frame, col).rename(columns={col: "level"})
        table = table.set_index("level").loc[order].reset_index()
        collector.add(table, "vehicle_mix", col, "2023-2024", path.name)
        ax = fig.add_subplot(grid[i])
        colors = [PALETTE.series[0] if lv != "Not involved" else PALETTE.ink_muted
                  for lv in table["level"]]
        dot_whisker(ax, table, reference=overall, color=colors)
        ax.set_xlim(0, 0.4)
        ax.set_title(title, pad=4)
        if i < len(groups) - 1:
            ax.tick_params(axis="x", labelbottom=False)
    add_titles(fig, "Pedestrians, motorcycles and single-vehicle crashes raise severity",
               f"Severe rate with 95% Wilson CI, 2023–2024 (n = {len(dev):,}). Vertical line = "
               f"overall rate {overall:.1%}; grey = comparison group.")
    return save_figure(fig, path)


def plot_hexbin_map(
    dev: pd.DataFrame,
    path: Path,
    collector: SummaryCollector,
    min_count: int = 40,
    gridsize: tuple[int, int] = (45, 40),
) -> Path:
    """Figure 7: hexbin map of severe rate relative to the overall rate.

    Args:
        dev: 2023-2024 rows.
        path: Output PNG.
        collector: Summary sink.
        min_count: Hexes with fewer collisions are drawn in neutral grey.
        gridsize: Hexagons in the (x, y) directions.

    Returns:
        Written path.
    """
    dev = dev_rows(dev).dropna(subset=["longitude", "latitude"])
    overall = float(dev["target"].mean())
    rural = rate_table(dev.assign(level=np.where(dev["latitude"] >= 55.8, "Scotland (lat ≥ 55.8)",
                                                 "England & Wales (lat < 55.8)")), "level")
    collector.add(rural, "geography", "latitude_band", "2023-2024", path.name)
    cmap = LinearSegmentedColormap.from_list("div", PALETTE.diverging)
    span = 0.2
    norm_ = TwoSlopeNorm(vcenter=overall, vmin=max(0.0, overall - span), vmax=overall + span)
    fig, ax = plt.subplots(figsize=(7.6, 9.6))
    fig.subplots_adjust(left=0.02, right=0.86, top=0.91, bottom=0.04)
    hb = ax.hexbin(dev["longitude"], dev["latitude"], C=dev["target"].astype(float),
                   reduce_C_function=np.mean, gridsize=gridsize, mincnt=min_count, cmap=cmap,
                   norm=norm_, linewidths=0.2, edgecolors=PALETTE.surface)
    sparse = ax.hexbin(dev["longitude"], dev["latitude"], gridsize=gridsize, mincnt=1,
                       linewidths=0.2, edgecolors=PALETTE.surface, zorder=0)
    counts = sparse.get_array()
    sparse.set_array(None)
    sparse.set_facecolor(PALETTE.grid)
    logger.info(f"Hexbin: {len(counts)} hexes, {int((counts >= min_count).sum())} drawn in colour")
    ax.set_aspect(1 / np.cos(np.deg2rad(54.5)))
    ax.set_axis_off()
    cbar = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.01, extend="both")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    cbar.outline.set_visible(False)
    cbar.set_label(f"Severe rate (centre = overall {overall:.1%})", color=PALETTE.ink_secondary)
    add_titles(fig, "Severity is lowest around London and highest in Scotland and the north",
               f"Severe rate per hexagon, 2023–2024. Coloured hexes have ≥ {min_count} "
               "collisions;\nlight grey = fewer. Red above, blue below the overall rate.")
    add_note(fig, "Caveat: rates mix road risk with force reporting practice (see police-force "
             "figure).")
    return save_figure(fig, path)


def force_reporting_group(dev: pd.DataFrame) -> pd.Series:
    """Classify each police force by its reporting system across the development years.

    Args:
        dev: 2023-2024 rows with ``police_force`` and ``collision_injury_based``.

    Returns:
        Series indexed by force code with one of three group labels.
    """
    share = dev.groupby(["police_force", "collision_year"])["collision_injury_based"].mean()
    share = share.unstack()
    lo, hi = share.min(axis=1), share.max(axis=1)
    return pd.Series(
        np.select([lo >= 0.9, hi <= 0.1], ["Injury-based both years", "Severity-based both years"],
                  "Switched / mixed 2023–24"),
        index=share.index,
    )


def plot_police_force_rank(
    dev: pd.DataFrame, path: Path, collector: SummaryCollector, codebook: Codebook,
    k: int = 10, min_n: int = 1000,
) -> Path:
    """Figure 8: top and bottom ``k`` police forces by severe rate, coloured by reporting system.

    Args:
        dev: 2023-2024 rows.
        path: Output PNG.
        collector: Summary sink.
        codebook: Data guide (force names).
        k: Forces shown at each end.
        min_n: Forces with fewer collisions are excluded from the ranking.

    Returns:
        Written path.
    """
    dev = dev_rows(dev)
    names = code_labeller(codebook, "collision", "police_force")
    groups = force_reporting_group(dev)
    table = rate_table(dev, "police_force", min_n=min_n).sort_values("rate", ascending=False)
    table["level"] = table["police_force"].map(names)
    table["group"] = table["police_force"].map(groups)
    collector.add(table, "geography", "police_force", "2023-2024", path.name)
    group_rates = rate_table(dev.assign(level=dev["police_force"].map(groups)), "level")
    collector.add(group_rates, "geography", "force_reporting_group", "2023-2024", path.name)
    shown = pd.concat([table.head(k), table.tail(k)], ignore_index=True)
    order = ["Injury-based both years", "Severity-based both years", "Switched / mixed 2023–24"]
    colours = dict(zip(order, PALETTE.series, strict=True))
    overall = float(dev["target"].mean())

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    fig.subplots_adjust(left=0.3, right=0.95, top=0.84, bottom=0.12)
    y = np.r_[np.arange(2 * k + 1)[::-1][:k], np.arange(2 * k + 1)[::-1][k + 1:]]
    ax.barh(y, shown["rate"], height=0.62, color=shown["group"].map(colours),
            edgecolor=PALETTE.surface, linewidth=1)
    ax.hlines(y, shown["ci_low"], shown["ci_high"], color=PALETTE.ink, linewidth=1.2)
    ax.axvline(overall, color=PALETTE.ink_muted, linewidth=1)
    ax.text(overall, 2 * k + 0.7, f"overall {overall:.1%}", ha="center", va="bottom",
            fontsize=8.5, color=PALETTE.ink_secondary)
    ax.text(0.005, k, f"…  {len(table) - 2 * k} forces not shown  …", ha="left", va="center",
            fontsize=9, color=PALETTE.ink_muted)
    ax.set_yticks(y, [f"{lv}  (n={n:,})" for lv, n in zip(shown["level"], shown["n"],
                                                          strict=True)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, 2 * k + 1.4)
    pct_axis(ax, "x")
    handles = [Patch(color=colours[g], label=g) for g in order]
    ax.legend(handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(-0.42, 1.02),
              handlelength=1.1)
    gr = group_rates.set_index("level")["rate"]
    add_titles(fig, "Force rankings largely track the recording system",
               f"Severe rate by police force, top and bottom {k} of {len(table)} forces with ≥ "
               f"{min_n:,} collisions, 2023–2024; whiskers = 95% Wilson CI.")
    add_note(fig, f"Pooled severe rate: injury-based forces {gr[order[0]]:.1%}, severity-based "
             f"{gr[order[1]]:.1%}, switched/mixed {gr[order[2]]:.1%}. Injury-based systems "
             "(CRaSH/COPA) classify\nmore casualties as serious, so a force's rank reflects "
             "recording practice as well as road risk.")
    return save_figure(fig, path)


def plot_reporting_system(
    dev: pd.DataFrame, path: Path, collector: SummaryCollector, codebook: Codebook
) -> Path:
    """Figure 9: severe rate for injury-based vs severity-based recording, per year.

    Args:
        dev: 2023-2024 rows.
        path: Output PNG.
        collector: Summary sink.
        codebook: Data guide.

    Returns:
        Written path.
    """
    dev = dev_rows(dev)
    labels = code_labeller(codebook, "collision", "collision_injury_based")
    table = rate_table(dev, ["collision_year", "collision_injury_based"])
    table["system"] = table["collision_injury_based"].map(labels)
    table["level"] = table["collision_year"].astype(str) + " · " + table["system"]
    table = table.sort_values(["collision_year", "collision_injury_based"],
                              ascending=[True, False]).reset_index(drop=True)
    collector.add(table, "reporting_system", "year_x_injury_based", "2023-2024", path.name)
    pooled = rate_table(dev, "collision_injury_based")
    pooled["level"] = pooled["collision_injury_based"].map(labels)
    collector.add(pooled, "reporting_system", "collision_injury_based", "2023-2024", path.name)
    colours = [PALETTE.series[0] if v == 1 else PALETTE.series[1]
               for v in table["collision_injury_based"]]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    fig.subplots_adjust(left=0.37, right=0.95, top=0.72, bottom=0.15)
    dot_whisker(ax, table, reference=float(dev["target"].mean()), color=colours)
    ax.set_xlim(0.1, 0.35)
    for yi, row in zip(np.arange(len(table))[::-1], table.itertuples(index=False), strict=True):
        ax.text(row.ci_high + 0.004, yi, f"{row.rate:.1%}", va="center", fontsize=9,
                color=PALETTE.ink)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=c, label=lab)
               for c, lab in ((PALETTE.series[0], labels[1]), (PALETTE.series[1], labels[0]))]
    ax.legend(handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(-0.55, 1.02))
    p = pooled.set_index("collision_injury_based")["rate"]
    add_titles(fig, "Injury-based recording yields a markedly higher severe rate",
               f"Severe rate by recording system and year, 2023–2024; pooled {p[1]:.1%} vs "
               f"{p[0]:.1%}. Whiskers = 95% Wilson CI.")
    return save_figure(fig, path)


def plot_police_attendance(
    dev: pd.DataFrame, path: Path, collector: SummaryCollector, codebook: Codebook
) -> Path:
    """Figure 10: severe rate by police attendance (leakage illustration).

    Args:
        dev: 2023-2024 rows.
        path: Output PNG.
        collector: Summary sink.
        codebook: Data guide.

    Returns:
        Written path.
    """
    dev = dev_rows(dev)
    labels = code_labeller(codebook, "collision", "did_police_officer_attend_scene_of_accident")
    labels = {**labels, 1: "Officer attended", 2: "Not attended"}
    table = rate_table(dev, "police_attended")
    table["level"] = label_levels(table["police_attended"], labels)
    collector.add(table, "leakage", "police_attended", "2023-2024", path.name)
    severe = dev.loc[dev["target"] == 1, "police_attended"].eq(1).mean()
    slight = dev.loc[dev["target"] == 0, "police_attended"].eq(1).mean()
    fig, ax = plt.subplots(figsize=(9, 3.3))
    fig.subplots_adjust(left=0.33, right=0.95, top=0.76, bottom=0.22)
    dot_whisker(ax, table, reference=float(dev["target"].mean()))
    ax.set_xlim(0, 0.4)
    for yi, row in zip(np.arange(len(table))[::-1], table.itertuples(index=False), strict=True):
        ax.text(row.ci_high + 0.006, yi, f"{row.rate:.1%}", va="center", fontsize=9,
                color=PALETTE.ink)
    add_titles(fig, "Police attendance follows severity — excluded as leakage",
               f"Severe rate by attendance, 2023–2024. An officer attended {severe:.0%} of "
               f"severe vs {slight:.0%} of slight collisions.")
    add_note(fig, "Attendance is decided after the outcome is known, so it is kept outside the "
             "feature set and used only as an evaluation stratifier.")
    return save_figure(fig, path)


def plot_missingness(
    table: pd.DataFrame, path: Path, palette: Palette = PALETTE
) -> Path:
    """Figure 11: missingness by field and year (label-free, includes 2025).

    Args:
        table: Frame indexed by display label with one column per year (shares in [0, 1]) and a
            ``group`` column naming the field block.
        path: Output PNG.
        palette: Colours.

    Returns:
        Written path.
    """
    years = [c for c in table.columns if c != "group"]
    values = table[years].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq_blue", ("#f4f8fd", *palette.sequential))
    n_rows = len(table)
    fig, ax = plt.subplots(figsize=(8.6, 0.33 * n_rows + 1.9))
    fig.subplots_adjust(left=0.4, right=0.76, top=1 - 1.0 / fig.get_figheight(),
                        bottom=0.35 / fig.get_figheight())
    ax.pcolormesh(np.arange(len(years) + 1), np.arange(n_rows + 1), values[::-1], cmap=cmap,
                  vmin=0, vmax=1, edgecolors=palette.surface, linewidth=2)
    for i in range(n_rows):
        for j in range(len(years)):
            v = values[i, j]
            ink = "#ffffff" if v > 0.45 else palette.ink
            ax.text(j + 0.5, n_rows - i - 0.5, f"{v:.0%}" if v >= 0.005 else "<1%",
                    ha="center", va="center", fontsize=8.5, color=ink)
    ax.set_xticks(np.arange(len(years)) + 0.5, [str(y) for y in years])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_rows) + 0.5, list(table.index)[::-1])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    boundaries = table["group"].ne(table["group"].shift()).to_numpy().nonzero()[0]
    for start in boundaries:
        group = table["group"].iloc[start]
        ax.text(len(years) + 0.15, n_rows - start - 0.5, group, fontsize=8.5, va="center",
                ha="left", color=palette.ink_secondary, weight="bold")
        if start > 0:
            ax.axhline(n_rows - start, color=palette.ink_muted, linewidth=1)
    add_titles(fig, "Schema drift: several fields are phased out by 2025",
               "Share of rows missing (−1 / unknown codes) by field and year. Label-free view; "
               "includes the 2025 test year.")
    return save_figure(fig, path)
