"""Model factories for the six model families compared in the study."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODEL_NAMES = ("lr", "dt", "rf", "xgb", "lgbm", "cat")
BOOSTERS = ("xgb", "lgbm", "cat")
GPU_MODELS = ("xgb", "cat")
MAX_BOOST_ROUNDS = 4000


def make_model(
    name: str,
    params: dict[str, Any],
    seed: int,
    device: str = "cpu",
    n_jobs: int = -1,
    n_estimators: int | None = None,
) -> Any:
    """Instantiate an unfitted classifier.

    Args:
        name: Registry key in :data:`MODEL_NAMES`.
        params: Hyperparameters (from Optuna or a frozen config).
        seed: Random seed.
        device: ``cpu`` or ``cuda`` (only used by XGBoost and CatBoost).
        n_jobs: CPU threads.
        n_estimators: Fixed number of boosting rounds for refits; ``None`` = early-stopping mode.

    Returns:
        A scikit-learn compatible estimator.
    """
    p = dict(params)
    if name == "lr":
        return LogisticRegression(
            C=p.get("C", 1.0), solver="lbfgs", max_iter=p.get("max_iter", 1000), tol=1e-4,
            random_state=seed,
        )
    if name == "dt":
        return DecisionTreeClassifier(random_state=seed, **p)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **p)
    rounds = n_estimators if n_estimators is not None else MAX_BOOST_ROUNDS
    if name == "xgb":
        return XGBClassifier(
            n_estimators=rounds, tree_method="hist", device=device, enable_categorical=True,
            max_cat_to_onehot=8, eval_metric="logloss", random_state=seed, n_jobs=n_jobs,
            early_stopping_rounds=None if n_estimators is not None else 100, **p,
        )
    if name == "lgbm":
        return LGBMClassifier(
            n_estimators=rounds, random_state=seed, n_jobs=n_jobs, verbose=-1, **p,
        )
    if name == "cat":
        task = "GPU" if device == "cuda" else "CPU"
        # Single-feature CTRs only: feature-combination CTRs were ~3x slower on GPU (see report).
        extra: dict[str, Any] = {"max_ctr_complexity": 1}
        if task == "GPU":
            extra |= {"devices": "0", "gpu_ram_part": 0.45}
        return CatBoostClassifier(
            iterations=rounds, random_seed=seed, task_type=task, thread_count=n_jobs,
            eval_metric="Logloss", verbose=0, allow_writing_files=False,
            od_type=None if n_estimators is not None else "Iter",
            od_wait=None if n_estimators is not None else 100, **extra, **p,
        )
    raise ValueError(f"Unknown model: {name}")


def best_iteration(model: Any, name: str) -> int | None:
    """Return the early-stopping best iteration (number of rounds) of a fitted booster.

    Args:
        model: Fitted estimator.
        name: Registry key.

    Returns:
        Best number of boosting rounds, or ``None`` for non-boosters.
    """
    if name == "xgb":
        return int(model.best_iteration) + 1
    if name == "lgbm":
        return int(model.best_iteration_ or model.n_estimators)
    if name == "cat":
        return int(model.get_best_iteration()) + 1
    return None


def fit_model(
    model: Any,
    name: str,
    X: Any,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    eval_set: tuple[Any, np.ndarray] | None = None,
    cat_features: list[str] | None = None,
) -> Any:
    """Fit a model, wiring early stopping and categorical handling per library.

    Args:
        model: Unfitted estimator from :func:`make_model`.
        name: Registry key.
        X: Training matrix.
        y: Training labels.
        sample_weight: Optional per-row weights.
        eval_set: Optional ``(X_val, y_val)`` for early stopping (boosters only).
        cat_features: Categorical column names (CatBoost only).

    Returns:
        The fitted model.
    """
    if name == "xgb":
        kw: dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            kw["eval_set"] = [eval_set]
        return model.fit(X, y, sample_weight=sample_weight, **kw)
    if name == "lgbm":
        import lightgbm as lgb

        if eval_set is None:
            return model.fit(X, y, sample_weight=sample_weight)
        return model.fit(
            X, y, sample_weight=sample_weight, eval_set=[eval_set], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    if name == "cat":
        X = _catboost_frame(X, cat_features or [])
        kw = {"cat_features": cat_features or []}
        if eval_set is not None:
            kw["eval_set"] = (_catboost_frame(eval_set[0], cat_features or []), eval_set[1])
            kw["use_best_model"] = True
        return model.fit(X, y, sample_weight=sample_weight, **kw)
    return model.fit(X, y, sample_weight=sample_weight)


def predict_proba(
    model: Any, name: str, X: Any, cat_features: list[str] | None = None
) -> np.ndarray:
    """Predict positive-class probabilities.

    Args:
        model: Fitted estimator.
        name: Registry key.
        X: Feature matrix in the model's representation.
        cat_features: Categorical column names (CatBoost only).

    Returns:
        1-D array of P(severe).
    """
    if name == "cat":
        X = _catboost_frame(X, cat_features or [])
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def _catboost_frame(X: Any, cat_features: list[str]) -> Any:
    """CatBoost requires categorical columns as strings/ints without NaN."""
    X = X.copy()
    for c in cat_features:
        X[c] = X[c].astype(object).where(X[c].notna(), "__missing__").astype(str)
    return X
