"""Optuna hyperparameter search spaces (identical across ablation cells of the same model)."""

from __future__ import annotations

from typing import Any

import optuna


def suggest_params(trial: optuna.Trial, name: str) -> dict[str, Any]:
    """Sample hyperparameters for a model family.

    Args:
        trial: Optuna trial.
        name: Registry key (``lr``, ``dt``, ``rf``, ``xgb``, ``lgbm``, ``cat``).

    Returns:
        Hyperparameter dict accepted by :func:`src.models.registry.make_model`.
    """
    if name == "lr":
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
    if name == "dt":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 1000, log=True),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-7, 1e-3, log=True),
        }
    if name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_int("max_depth", 8, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100, log=True),
            "max_features": trial.suggest_float("max_features", 0.05, 0.3),
            "max_samples": trial.suggest_float("max_samples", 0.3, 0.8),
        }
    if name == "xgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        }
    if name == "lgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        }
    if name == "cat":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    raise ValueError(f"Unknown model: {name}")
