"""Feature selection for F1 (plan section 3), run on the FIT rows of ``F1full_tune`` only.

Stages (each stage only sees survivors of the previous one):

1. near-constant: the most frequent value (NaN counted as a value) covers > 99.5% of rows.
2. redundancy: |Spearman| between numeric pairs or Cramér's V between categorical pairs > 0.95;
   of each redundant pair the feature with lower mutual information with the target is dropped
   (greedy, strongest pairs first; ties keep the earlier column, i.e. the F0 one).
3. permutation importance: LightGBM, leave-one-quarter-out CV on 2023 (train on 3 quarters,
   permute each feature on the held-out quarter, ``n_repeats`` repeats, score = drop in PR-AUC);
   keep features whose mean importance is positive in at least ``min_positive_folds`` folds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from scipy.stats.contingency import association
from sklearn.metrics import average_precision_score, mutual_info_score

from src.paths import CONFIGS_DIR, TABLES_DIR

logger = logging.getLogger(__name__)

SELECTED_YAML_PATH: Path = CONFIGS_DIR / "features" / "f1_selected.yaml"
SELECTION_TABLE_PATH: Path = TABLES_DIR / "feature_selection.csv"
QUARTER_SOURCE_COL = "month"
DEFAULT_LGBM_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


@dataclass(frozen=True)
class SelectionConfig:
    """Settings of the selection pipeline.

    Attributes:
        near_constant_share: Top-value share above which a feature is near-constant.
        redundancy_threshold: |Spearman| / Cramér's V above which a pair is redundant.
        mi_bins: Quantile bins used to discretise numeric features for mutual information.
        n_repeats: Permutation repeats per feature and fold.
        min_positive_folds: Folds with positive mean importance required to keep a feature.
        n_jobs: LightGBM threads (<= 12).
        seed: Seed of LightGBM and the permutations.
        lgbm_params: LightGBM parameters of the importance model.
    """

    near_constant_share: float = 0.995
    redundancy_threshold: float = 0.95
    mi_bins: int = 32
    n_repeats: int = 5
    min_positive_folds: int = 3
    n_jobs: int = 12
    seed: int = 42
    lgbm_params: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))


def is_categorical(series: pd.Series) -> bool:
    """Return True for ``category`` dtype."""
    return isinstance(series.dtype, pd.CategoricalDtype)


# --- stage 1 ----------------------------------------------------------------------------------
def top_value_share(series: pd.Series) -> float:
    """Share of the most frequent value (NaN counted as a value)."""
    return float(series.value_counts(dropna=False, normalize=True).iloc[0])


def top_value_shares(X: pd.DataFrame) -> pd.Series:
    """Return the top-value share of every column.

    Args:
        X: Feature frame.

    Returns:
        Float series indexed by column.
    """
    return pd.Series({c: top_value_share(X[c]) for c in X.columns}, dtype="float64")


# --- mutual information -----------------------------------------------------------------------
def discretise(series: pd.Series, n_bins: int) -> np.ndarray:
    """Integer codes for MI: categories / low-cardinality values as-is, else quantile bins.

    Missing values get their own code (-1).

    Args:
        series: Feature values.
        n_bins: Quantile bins for high-cardinality numerics.

    Returns:
        int64 code array.
    """
    if is_categorical(series):
        return series.cat.codes.to_numpy().astype(np.int64)
    values = series.astype("float64")
    if values.nunique() > n_bins:
        values = pd.qcut(values, q=n_bins, labels=False, duplicates="drop")
    codes = pd.factorize(values, use_na_sentinel=True)[0]
    return codes.astype(np.int64)


def mutual_information(X: pd.DataFrame, y: np.ndarray, n_bins: int) -> pd.Series:
    """Mutual information (nats) between each discretised feature and the target.

    Args:
        X: Feature frame.
        y: Binary labels.
        n_bins: Quantile bins for numerics.

    Returns:
        Float series indexed by column.
    """
    return pd.Series({c: mutual_info_score(y, discretise(X[c], n_bins)) for c in X.columns},
                     dtype="float64")


# --- stage 2 ----------------------------------------------------------------------------------
def spearman_pairs(X: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Numeric pairs with |Spearman| above ``threshold``.

    Ranks are computed per column (NaN kept) and correlated with pairwise-complete Pearson.

    Args:
        X: Numeric feature frame.
        threshold: Absolute-correlation threshold.

    Returns:
        List of ``(a, b, |rho|)`` with ``a`` before ``b`` in column order.
    """
    ranks = X.rank(method="average")
    corr = ranks.corr(method="pearson", min_periods=100).abs().to_numpy()
    cols = list(X.columns)
    i_idx, j_idx = np.triu_indices(len(cols), k=1)
    vals = corr[i_idx, j_idx]
    hit = np.nan_to_num(vals, nan=0.0) > threshold
    pairs = zip(i_idx[hit], j_idx[hit], vals[hit], strict=True)
    return [(cols[i], cols[j], float(v)) for i, j, v in pairs]


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V between two categoricals (missing treated as its own level).

    Args:
        a: First categorical.
        b: Second categorical.

    Returns:
        V in [0, 1] (0 if either has a single level).
    """
    table = pd.crosstab(a.cat.codes, b.cat.codes).to_numpy()
    if min(table.shape) < 2:
        return 0.0
    return float(association(table, method="cramer"))


def cramers_pairs(X: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Categorical pairs with Cramér's V above ``threshold``.

    Args:
        X: Categorical feature frame.
        threshold: V threshold.

    Returns:
        List of ``(a, b, V)`` with ``a`` before ``b`` in column order.
    """
    cols = list(X.columns)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = cramers_v(X[a], X[b])
            if v > threshold:
                pairs.append((a, b, v))
    return pairs


def resolve_redundancy(pairs: list[tuple[str, str, float]], mi: pd.Series
                       ) -> dict[str, tuple[str, float]]:
    """Greedy pruning: strongest pairs first, drop the lower-MI member if both survive.

    Args:
        pairs: ``(a, b, strength)`` with ``a`` earlier in column order.
        mi: Mutual information per feature.

    Returns:
        ``{dropped feature: (kept partner, strength)}``.
    """
    dropped: dict[str, tuple[str, float]] = {}
    for a, b, strength in sorted(pairs, key=lambda p: -p[2]):
        if a in dropped or b in dropped:
            continue
        loser, winner = (a, b) if mi[a] < mi[b] else (b, a)
        dropped[loser] = (winner, strength)
    return dropped


# --- stage 3 ----------------------------------------------------------------------------------
def quarter_of(X: pd.DataFrame) -> np.ndarray:
    """Calendar quarter (1-4) of each row from the ``month`` feature (taken from the full
    candidate frame, so it works even if ``month`` itself is pruned)."""
    month = X[QUARTER_SOURCE_COL].to_numpy(dtype=np.float64)
    if np.isnan(month).any():
        raise ValueError("Missing month; cannot assign quarters")
    return ((month - 1) // 3 + 1).astype(int)


def permute_column(X: pd.DataFrame, col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Return a shallow copy of ``X`` with one column shuffled (dtype preserved)."""
    out = X.copy(deep=False)
    perm = rng.permutation(len(X))
    out[col] = X[col].iloc[perm].set_axis(X.index)
    return out


def fold_importance(X_tr: pd.DataFrame, y_tr: np.ndarray, X_ho: pd.DataFrame, y_ho: np.ndarray,
                    config: SelectionConfig, fold_seed: int) -> tuple[pd.DataFrame, float]:
    """Fit LightGBM on training quarters and permute each feature on the held-out quarter.

    Args:
        X_tr: Training-quarter features.
        y_tr: Training labels.
        X_ho: Held-out-quarter features.
        y_ho: Held-out labels.
        config: Selection configuration.
        fold_seed: Seed of this fold's permutations.

    Returns:
        ``(frame of PR-AUC drops [features x repeats], baseline PR-AUC)``.
    """
    model = lgb.LGBMClassifier(**config.lgbm_params, n_jobs=config.n_jobs,
                               random_state=config.seed)
    model.fit(X_tr, y_tr)
    base = average_precision_score(y_ho, model.predict_proba(X_ho)[:, 1])
    rng = np.random.default_rng(fold_seed)
    drops = np.zeros((X_ho.shape[1], config.n_repeats))
    for i, col in enumerate(X_ho.columns):
        for r in range(config.n_repeats):
            p = model.predict_proba(permute_column(X_ho, col, rng))[:, 1]
            drops[i, r] = base - average_precision_score(y_ho, p)
    return pd.DataFrame(drops, index=X_ho.columns), float(base)


def permutation_importance_loqo(X: pd.DataFrame, y: np.ndarray, quarters: np.ndarray,
                                config: SelectionConfig) -> pd.DataFrame:
    """Leave-one-quarter-out permutation importance (mean PR-AUC drop per fold).

    Args:
        X: FIT-row features.
        y: Labels.
        quarters: Calendar quarter (1-4) of each row.
        config: Selection configuration.

    Returns:
        Frame indexed by feature with columns ``imp_q1..imp_q4`` (mean drop) and
        ``sd_q1..sd_q4`` (sd over repeats).
    """
    result = pd.DataFrame(index=X.columns)
    for q in sorted(np.unique(quarters)):
        t0 = time.time()
        ho = quarters == q
        drops, base = fold_importance(X[~ho], y[~ho], X[ho], y[ho], config, config.seed + int(q))
        result[f"imp_q{q}"] = drops.mean(axis=1)
        result[f"sd_q{q}"] = drops.std(axis=1, ddof=1)
        n_pos = int((drops.mean(axis=1) > 0).sum())
        logger.info(f"Q{q} held out: base PR-AUC {base:.4f}, {n_pos}/{len(drops)} features "
                    f"positive ({time.time() - t0:.0f}s)")
    return result


# --- orchestration ----------------------------------------------------------------------------
@dataclass
class SelectionResult:
    """Outcome of the selection.

    Attributes:
        selected: Selected features in original column order.
        table: Per-feature diagnostics (one row per candidate).
    """

    selected: list[str]
    table: pd.DataFrame


def _stage_reasons(table: pd.DataFrame, redundant: dict[str, tuple[str, float]],
                   config: SelectionConfig) -> pd.Series:
    """Compose one human-readable drop reason per feature ("" if selected)."""
    reasons = pd.Series("", index=table.index, dtype=object)
    nc = table["near_constant"]
    reasons[nc] = [f"near-constant (top value share {s:.4f} > {config.near_constant_share})"
                   for s in table.loc[nc, "top_share"]]
    for feat, (partner, strength) in redundant.items():
        reasons[feat] = (f"redundant with {partner} ({table.loc[feat, 'redundancy_metric']} "
                         f"{strength:.3f} > {config.redundancy_threshold}; lower MI)")
    low = table["stage"].eq("importance")
    reasons[low] = [f"permutation importance positive in {int(n)}/4 folds "
                    f"(< {config.min_positive_folds})" for n in table.loc[low, "n_positive_folds"]]
    return reasons


def select_features(X: pd.DataFrame, y: np.ndarray, config: SelectionConfig) -> SelectionResult:
    """Run the three selection stages on FIT rows.

    Args:
        X: FIT-row features of ``F1full_tune``.
        y: FIT labels.
        config: Selection configuration.

    Returns:
        :class:`SelectionResult`.
    """
    table = pd.DataFrame(index=pd.Index(X.columns, name="feature"))
    table["kind"] = ["categorical" if is_categorical(X[c]) else "numeric" for c in X.columns]
    table["top_share"] = top_value_shares(X)
    table["near_constant"] = table["top_share"] > config.near_constant_share
    table["stage"] = np.where(table["near_constant"], "near_constant", "")
    logger.info(f"Stage 1 near-constant: {int(table['near_constant'].sum())} dropped")

    t0 = time.time()
    table["mutual_info"] = mutual_information(X, y, config.mi_bins)
    alive = [c for c in X.columns if not table.loc[c, "near_constant"]]
    num = [c for c in alive if table.loc[c, "kind"] == "numeric"]
    cat = [c for c in alive if table.loc[c, "kind"] == "categorical"]
    pairs_num = spearman_pairs(X[num], config.redundancy_threshold)
    pairs_cat = cramers_pairs(X[cat], config.redundancy_threshold)
    redundant = resolve_redundancy(pairs_num + pairs_cat, table["mutual_info"])
    table["redundancy_metric"] = np.where(table["kind"] == "numeric", "|spearman|", "cramers_v")
    table["redundant_with"] = pd.Series({k: v[0] for k, v in redundant.items()}, dtype=object)
    table["redundancy_strength"] = pd.Series({k: v[1] for k, v in redundant.items()},
                                             dtype="float64")
    table.loc[list(redundant), "stage"] = "redundancy"
    logger.info(f"Stage 2 redundancy: {len(pairs_num)} numeric + {len(pairs_cat)} categorical "
                f"pairs above {config.redundancy_threshold}; {len(redundant)} dropped "
                f"({time.time() - t0:.0f}s)")

    survivors = [c for c in alive if c not in redundant]
    imp = permutation_importance_loqo(X[survivors], y, quarter_of(X), config)
    table = table.join(imp)
    imp_cols = [c for c in imp.columns if c.startswith("imp_q")]
    table["n_positive_folds"] = (table[imp_cols] > 0).sum(axis=1).where(
        table.index.isin(survivors))
    table["mean_importance"] = table[imp_cols].mean(axis=1)
    keep = table["n_positive_folds"] >= config.min_positive_folds
    table.loc[table.index.isin(survivors) & ~keep, "stage"] = "importance"
    table["selected"] = keep & table.index.isin(survivors)
    table["reason"] = _stage_reasons(table, redundant, config)
    selected = [c for c in X.columns if table.loc[c, "selected"]]
    logger.info(f"Stage 3 permutation importance: {len(survivors) - len(selected)} dropped; "
                f"{len(selected)}/{X.shape[1]} features selected")
    return SelectionResult(selected=selected, table=table)


def write_selection(result: SelectionResult, config: SelectionConfig, source: str,
                    yaml_path: Path = SELECTED_YAML_PATH,
                    table_path: Path = SELECTION_TABLE_PATH) -> None:
    """Write the frozen feature list (YAML) and the per-feature diagnostics (CSV).

    Args:
        result: Selection outcome.
        config: Selection configuration (recorded in the YAML).
        source: Description of the data the selection was run on.
        yaml_path: Output YAML path.
        table_path: Output CSV path.
    """
    table = result.table
    dropped = {f: str(table.loc[f, "reason"]) for f in table.index if not table.loc[f, "selected"]}
    payload = {
        "source": source,
        "n_candidates": len(table),
        "n_selected": len(result.selected),
        "settings": {k: v for k, v in asdict(config).items()},
        "selected": list(result.selected),
        "dropped": dropped,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=100))
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(table_path, index=False, float_format="%.6g")
    logger.info(f"Wrote {yaml_path} and {table_path}")


def load_selected(path: Path = SELECTED_YAML_PATH) -> list[str]:
    """Read the frozen F1 feature list.

    Args:
        path: ``f1_selected.yaml``.

    Returns:
        Selected feature names.
    """
    return list(yaml.safe_load(path.read_text())["selected"])
