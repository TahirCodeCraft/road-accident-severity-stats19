"""Run one ablation cell end to end: Optuna HPO on FIT->TUNE, then multi-seed refit on FIT+TUNE.

Feature-frame contract (written by ``scripts/03_build_features.py``)::

    data/processed/features/{feature_set}_{phase}.parquet
        phase = "tune":  encoders fitted on FIT;       rows of splits fit / tune / cal
        phase = "refit": encoders fitted on FIT+TUNE;  rows of splits fit / tune / cal / test
        columns: collision_index, split, target, <features...> (categoricals as ``category``)

Outputs (``experiments/<cell_id>/``): ``config.json``, ``optuna_trials.csv``, ``best_params.json``,
``model_seed{s}.joblib`` (model + adapter) and ``preds_cal_seed{s}.parquet``.
TEST rows are never read here (see ``scripts/08_evaluate_test.py``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score

from src.features.adapters import FeatureAdapter, make_adapter
from src.models.imbalance import apply_imbalance
from src.models.registry import BOOSTERS, best_iteration, fit_model, make_model, predict_proba
from src.models.search_spaces import suggest_params
from src.paths import EXPERIMENTS_DIR, PROCESSED_DIR
from src.utils.seed import set_seed
from src.utils.wandb_utils import finish, init_run, log_metrics

logger = logging.getLogger(__name__)

ID_COLS = ("collision_index", "split", "target")


@dataclass
class CellSpec:
    """One cell of the ablation matrix.

    Attributes:
        model: Registry key.
        feature_set: ``F0``, ``F1`` (selected) or ``F1full``.
        cs: Imbalance strategy (``none``, ``weight``, ``rus``, ``smotenc``).
        n_trials: Optuna trials.
        seeds: Refit seeds.
        device: ``cpu`` or ``cuda``.
        n_jobs: CPU threads per fit.
        hpo_seed: Optuna sampler seed.
        drop_features: Optional feature names removed (sensitivity analyses).
        tag: Optional suffix for the cell id (sensitivity analyses).
        feature_columns: Optional explicit feature list (overrides the frame's columns).
    """

    model: str
    feature_set: str
    cs: str
    n_trials: int = 30
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    device: str = "cpu"
    n_jobs: int = -1
    hpo_seed: int = 42
    drop_features: list[str] = field(default_factory=list)
    tag: str = ""
    feature_columns: list[str] | None = None

    @property
    def cell_id(self) -> str:
        """Unique, filesystem-friendly identifier."""
        suffix = f"__{self.tag}" if self.tag else ""
        return f"{self.feature_set}__{self.cs}__{self.model}{suffix}"


def feature_path(feature_set: str, phase: str) -> Path:
    """Path of a feature parquet.

    Args:
        feature_set: Feature-set name.
        phase: ``tune`` or ``refit``.

    Returns:
        Parquet path.
    """
    return PROCESSED_DIR / "features" / f"{feature_set}_{phase}.parquet"


def load_frames(spec: CellSpec, phase: str, splits: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load the requested (non-test) splits of a feature frame.

    Args:
        spec: Cell specification.
        phase: ``tune`` or ``refit``.
        splits: Split names to load; ``test`` is forbidden here.

    Returns:
        Mapping split name -> frame.
    """
    if "test" in splits:
        raise PermissionError("TEST rows may only be read by the frozen evaluation script.")
    table = ds.dataset(feature_path(spec.feature_set, phase)).to_table(
        filter=ds.field("split").isin(list(splits))
    )
    df = table.to_pandas()
    feats = spec.feature_columns or [c for c in df.columns if c not in ID_COLS]
    feats = [c for c in feats if c not in set(spec.drop_features)]
    df = df[list(ID_COLS) + feats]
    return {s: df[df["split"] == s].reset_index(drop=True) for s in splits}


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a frame into features and labels.

    Args:
        df: Frame with ID columns.

    Returns:
        ``(X, y)``.
    """
    return df.drop(columns=list(ID_COLS)), df["target"].to_numpy().astype(int)


def _fit_cell_model(
    spec: CellSpec,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    eval_df: pd.DataFrame | None = None,
    n_estimators: int | None = None,
) -> tuple[Any, FeatureAdapter, int | None]:
    """Apply the imbalance strategy, adapt features and fit one model.

    Returns:
        ``(model, adapter, best_iteration)``.
    """
    imb = apply_imbalance(X_train, y_train, spec.cs, seed)
    adapter = make_adapter(spec.model)
    adapter.num_fill = imb.num_fill  # SMOTE-NC: impute every split with the training medians
    X_fit = adapter.fit_transform(imb.X)
    model = make_model(spec.model, params, seed, spec.device, spec.n_jobs, n_estimators)
    if spec.model in ("lr", "dt", "rf") and spec.cs == "weight":
        model.set_params(class_weight={0: 1.0, 1: imb.pos_weight})
        sample_weight = None
    else:
        sample_weight = imb.sample_weight
    eval_set = None
    if eval_df is not None and spec.model in BOOSTERS:
        X_ev, y_ev = xy(eval_df)
        eval_set = (adapter.transform(X_ev), y_ev)
    model = fit_model(model, spec.model, X_fit, imb.y, sample_weight, eval_set, adapter.cat_cols)
    return model, adapter, best_iteration(model, spec.model) if eval_set else None


def _predict(spec: CellSpec, model: Any, adapter: FeatureAdapter, df: pd.DataFrame) -> np.ndarray:
    """Predict P(severe) for a frame."""
    X, _ = xy(df)
    return predict_proba(model, spec.model, adapter.transform(X), adapter.cat_cols)


def tune(spec: CellSpec, out_dir: Path) -> tuple[dict[str, Any], int | None]:
    """Optuna search maximising PR-AUC on TUNE (models fitted on FIT).

    Args:
        spec: Cell specification.
        out_dir: Cell output directory.

    Returns:
        ``(best_params, best_iteration)``.
    """
    frames = load_frames(spec, "tune", ("fit", "tune"))
    X_fit, y_fit = xy(frames["fit"])
    y_tune = frames["tune"]["target"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, spec.model)
        model, adapter, n_iter = _fit_cell_model(
            spec, params, X_fit, y_fit, spec.hpo_seed, eval_df=frames["tune"]
        )
        trial.set_user_attr("best_iteration", n_iter)
        return float(average_precision_score(y_tune, _predict(spec, model, adapter,
                                                                frames["tune"])))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=spec.hpo_seed))
    study.optimize(objective, n_trials=spec.n_trials, gc_after_trial=True)
    study.trials_dataframe().to_csv(out_dir / "optuna_trials.csv", index=False)
    best = study.best_trial
    params = suggest_params(optuna.trial.FixedTrial(best.params), spec.model)
    logger.info("[%s] best TUNE PR-AUC %.4f params=%s", spec.cell_id, best.value, params)
    return params, best.user_attrs.get("best_iteration")


def refit(spec: CellSpec, params: dict[str, Any], best_iter: int | None, out_dir: Path) -> dict:
    """Refit on FIT+TUNE with frozen hyperparameters for each seed; predict CAL.

    Args:
        spec: Cell specification.
        params: Frozen hyperparameters.
        best_iter: Early-stopping rounds from HPO (boosters).
        out_dir: Cell output directory.

    Returns:
        Summary dict with CAL PR-AUC per seed.
    """
    frames = load_frames(spec, "refit", ("fit", "tune", "cal"))
    train = pd.concat([frames["fit"], frames["tune"]], ignore_index=True)
    X_tr, y_tr = xy(train)
    n_est = int(round(1.2 * best_iter)) if best_iter else None
    summary: dict[str, Any] = {"n_estimators": n_est, "cal_pr_auc": {}}
    for seed in spec.seeds:
        set_seed(seed)
        t0 = time.time()
        model, adapter, _ = _fit_cell_model(spec, params, X_tr, y_tr, seed, n_estimators=n_est)
        p_cal = _predict(spec, model, adapter, frames["cal"])
        joblib.dump({"model": model, "adapter": adapter, "spec": asdict(spec), "params": params},
                    out_dir / f"model_seed{seed}.joblib", compress=3)
        pd.DataFrame({"collision_index": frames["cal"]["collision_index"],
                      "target": frames["cal"]["target"], "prob": p_cal}).to_parquet(
            out_dir / f"preds_cal_seed{seed}.parquet", index=False)
        ap = float(average_precision_score(frames["cal"]["target"], p_cal))
        summary["cal_pr_auc"][seed] = ap
        logger.info("[%s] seed %d refit %.0fs CAL PR-AUC %.4f", spec.cell_id, seed,
                    time.time() - t0, ap)
    return summary


def run_cell(spec: CellSpec, overwrite: bool = False, wandb_project: str = "dstt-severity") -> Path:
    """Run HPO + refit for one cell, saving config and artefacts.

    Args:
        spec: Cell specification.
        overwrite: Re-run even if a completed run exists.
        wandb_project: wandb project name (offline by default).

    Returns:
        Cell output directory.
    """
    out_dir = EXPERIMENTS_DIR / "cells" / spec.cell_id
    done = out_dir / "summary.json"
    if done.exists() and not overwrite:
        logger.info("[%s] already complete, skipping", spec.cell_id)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(spec), indent=2))
    run = init_run(project=wandb_project, config=asdict(spec), name=spec.cell_id,
                   group=spec.feature_set)
    t0 = time.time()
    params, best_iter = tune(spec, out_dir)
    (out_dir / "best_params.json").write_text(
        json.dumps({"params": params, "best_iteration": best_iter}, indent=2))
    summary = refit(spec, params, best_iter, out_dir)
    summary.update({"cell_id": spec.cell_id, "minutes": (time.time() - t0) / 60,
                    "params": params})
    log_metrics(run, {"cal_pr_auc_mean": float(np.mean(list(summary["cal_pr_auc"].values())))})
    finish(run)
    done.write_text(json.dumps(summary, indent=2, default=str))
    return out_dir
