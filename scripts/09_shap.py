"""SHAP explanations for the selected (frozen) model on CAL and TEST samples.

Outputs (reports/figures/shap_*.png, reports/tables/shap_*.csv):
    beeswarm + bar (TEST), dependence plots for the top features, local waterfalls for a true
    positive, a false negative and a false positive, and the CAL-vs-TEST rank stability of mean
    |SHAP| (Spearman).
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import sys
from pathlib import Path as _Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.posthoc import load_decision, require_frozen_manifest  # noqa: E402
from src.models.registry import predict_proba  # noqa: E402
from src.paths import EXPERIMENTS_DIR, FIGURES_DIR, TABLES_DIR  # noqa: E402
from src.training.runner import feature_path  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)
N_SAMPLE = 20_000


def shap_values(model: Any, name: str, X: pd.DataFrame, cat_cols: list[str]) -> tuple[np.ndarray,
                                                                                      float]:
    """Exact TreeSHAP contributions in log-odds space using each library's native routine.

    Args:
        model: Fitted model.
        name: Registry key.
        X: Adapted feature matrix.
        cat_cols: Categorical columns (CatBoost).

    Returns:
        ``(shap matrix [n, p], expected value)``.
    """
    if name == "lgbm":
        contrib = model.predict(X, pred_contrib=True)
    elif name == "xgb":
        import xgboost as xgb

        dm = xgb.DMatrix(X, enable_categorical=True)
        contrib = model.get_booster().predict(dm, pred_contribs=True)
    elif name == "cat":
        from catboost import Pool

        Xc = X.copy()
        for c in cat_cols:
            Xc[c] = Xc[c].astype(object).where(Xc[c].notna(), "__missing__").astype(str)
        contrib = model.get_feature_importance(Pool(Xc, cat_features=cat_cols), type="ShapValues")
    else:
        raise ValueError(f"Exact log-odds TreeSHAP is only wired for boosters, not {name}")
    contrib = np.asarray(contrib)
    return contrib[:, :-1], float(contrib[0, -1])


BOOSTER_KEYS = ("xgb", "lgbm", "cat")


def best_booster_cell() -> str:
    """Best booster cell (untagged) by mean raw CAL PR-AUC from the validation table.

    Returns:
        Cell id.
    """
    val = pd.read_csv(TABLES_DIR / "validation_results.csv")
    v = val[(val["tag"].fillna("") == "") & (val["variant"] == "raw") & (val["op"] == "t050")
            & val["model"].isin(BOOSTER_KEYS)]
    return str(v.groupby("cell_id")["pr_auc"].mean().idxmax())


def sample_split(feature_set: str, split: str, seed: int) -> pd.DataFrame:
    """Random sample of a split from the refit-phase parquet.

    Args:
        feature_set: Feature-set name.
        split: ``cal`` or ``test``.
        seed: Sampling seed.

    Returns:
        Sampled frame.
    """
    if split == "test":
        require_frozen_manifest()
    df = ds.dataset(feature_path(feature_set, "refit")).to_table(
        filter=ds.field("split") == split).to_pandas()
    return df.sample(n=min(N_SAMPLE, len(df)), random_state=seed).reset_index(drop=True)


def explanation(values: np.ndarray, base: float, X: pd.DataFrame) -> shap.Explanation:
    """Build a shap Explanation with numeric display data (categoricals as codes).

    Args:
        values: SHAP matrix.
        base: Expected value.
        X: Feature frame (pre-adaptation display data).

    Returns:
        Explanation object.
    """
    disp = X.copy()
    for c in disp.columns:
        if isinstance(disp[c].dtype, pd.CategoricalDtype):
            disp[c] = pd.to_numeric(disp[c].astype(object), errors="coerce")
    return shap.Explanation(values=values, base_values=np.full(len(X), base),
                            data=disp.to_numpy(dtype=float), feature_names=list(X.columns))


def save(fig_name: str) -> None:
    """Save and close the current matplotlib figure.

    Args:
        fig_name: File name inside the figures directory.
    """
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / fig_name, dpi=200, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default=None, help="Cell id (default: frozen selection)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed)
    manifest = require_frozen_manifest()
    cell_id = args.cell or manifest["selected"]["cell_id"]
    if cell_id.split("__")[2] not in BOOSTER_KEYS:
        cell_id = best_booster_cell()
        logger.info("Selected model is not a booster; explaining best booster cell %s", cell_id)
    cell_dir = EXPERIMENTS_DIR / "cells" / cell_id
    spec = json.loads((cell_dir / "config.json").read_text())
    bundle = joblib.load(cell_dir / f"model_seed{args.seed}.joblib")
    model, adapter = bundle["model"], bundle["adapter"]
    decision, cals = load_decision(cell_dir, args.seed)
    v = decision.best_calibrator
    thr = decision.thresholds[v]["tf2"]
    importance = {}
    for split in ("cal", "test"):
        df = sample_split(spec["feature_set"], split, args.seed)
        X = df[adapter.cat_cols + adapter.num_cols]
        Xa = adapter.transform(X)
        vals, base = shap_values(model, spec["model"], Xa, adapter.cat_cols)
        importance[split] = pd.Series(np.abs(vals).mean(axis=0), index=X.columns)
        if split != "test":
            continue
        exp = explanation(vals, base, X)
        shap.plots.beeswarm(exp, max_display=20, show=False)
        save("shap_beeswarm_test.png")
        shap.plots.bar(exp, max_display=20, show=False)
        save("shap_bar_test.png")
        top = importance["test"].sort_values(ascending=False).index[:4]
        for f in top:
            shap.plots.scatter(exp[:, f], color=exp, show=False)
            save(f"shap_dependence_{f}.png")
        p_raw = predict_proba(model, spec["model"], Xa, adapter.cat_cols)
        margin = vals.sum(axis=1) + base
        assert np.allclose(1 / (1 + np.exp(-margin)), p_raw, atol=1e-4), "SHAP not additive"
        p = np.clip(cals[v].predict(p_raw), 0, 1)
        y = df["target"].to_numpy()
        pred = p >= thr
        cases = {"tp": np.where(pred & (y == 1))[0], "fn": np.where(~pred & (y == 1))[0],
                 "fp": np.where(pred & (y == 0))[0]}
        for k, idx in cases.items():
            if len(idx):
                i = idx[np.argmax(p[idx])] if k != "fn" else idx[np.argmin(p[idx])]
                shap.plots.waterfall(exp[i], max_display=12, show=False)
                save(f"shap_waterfall_{k}.png")
    imp = pd.DataFrame({"mean_abs_shap_cal": importance["cal"],
                        "mean_abs_shap_test": importance["test"]}).sort_values(
        "mean_abs_shap_test", ascending=False)
    rho = spearmanr(imp["mean_abs_shap_cal"], imp["mean_abs_shap_test"]).statistic
    imp["rank_test"] = np.arange(1, len(imp) + 1)
    imp.to_csv(TABLES_DIR / "shap_importance.csv")
    (TABLES_DIR / "shap_stability.json").write_text(json.dumps(
        {"cell_id": cell_id, "spearman_cal_vs_test": float(rho), "n_sample": N_SAMPLE}, indent=2))
    logger.info("SHAP done for %s; CAL-vs-TEST importance Spearman %.3f", cell_id, rho)


if __name__ == "__main__":
    main()
