"""Tests for adapters, imbalance strategies and the model registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.adapters import FeatureAdapter, make_adapter
from src.models.imbalance import apply_imbalance, positive_weight
from src.models.registry import MODEL_NAMES, fit_model, make_model, predict_proba


def _toy_frame(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Mixed-type frame with NaNs and a learnable signal, ~25% positives."""
    rng = np.random.default_rng(seed)
    road = rng.choice([1, 2, 3, 6], size=n)
    speed = rng.choice([20.0, 30.0, 60.0, 70.0, np.nan], size=n)
    logit = -1.6 + 0.04 * np.nan_to_num(speed, nan=30.0) - 1.0 + (road == 6) * 0.8
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(
        {
            "road_type": pd.Categorical(road, categories=[1, 2, 3, 6]),
            "light": pd.Categorical(rng.choice([1, 4, np.nan], size=n), categories=[1.0, 4.0]),
            "speed_limit": speed.astype(np.float32),
            "n_veh": rng.integers(1, 4, size=n).astype(np.float32),
        }
    )
    return X, y


def test_positive_weight() -> None:
    assert positive_weight(np.array([0, 0, 0, 1])) == pytest.approx(3.0)


@pytest.mark.parametrize("strategy", ["none", "weight", "rus", "smotenc"])
def test_imbalance_strategies(strategy: str) -> None:
    X, y = _toy_frame()
    res = apply_imbalance(X, y, strategy, seed=0)
    assert list(res.X.columns) == list(X.columns)
    if strategy in ("rus", "smotenc"):
        assert res.y.mean() == pytest.approx(0.5, abs=0.01)
        assert isinstance(res.X["road_type"].dtype, pd.CategoricalDtype)
        assert list(res.X["road_type"].cat.categories) == [1, 2, 3, 6]
    if strategy == "weight":
        assert res.sample_weight is not None
        assert res.sample_weight[y == 1][0] == pytest.approx(positive_weight(y))
    if strategy == "rus":
        assert len(res.y) == 2 * int(y.sum())


def test_imbalance_does_not_mutate_input() -> None:
    X, y = _toy_frame()
    X_copy = X.copy()
    apply_imbalance(X, y, "smotenc", seed=0)
    pd.testing.assert_frame_equal(X, X_copy)


@pytest.mark.parametrize("kind", ["onehot", "ordinal", "native"])
def test_adapter_handles_unseen_and_missing(kind: str) -> None:
    X, _ = _toy_frame()
    adapter = FeatureAdapter(kind=kind, min_frequency=5).fit(X)
    X_new = X.head(5).copy()
    X_new["road_type"] = pd.Categorical([9, 9, np.nan, 1, 2])  # 9 unseen
    out = adapter.transform(X_new)
    assert len(out) == 5
    if kind == "ordinal":
        assert (out["road_type"].iloc[:3] == -1).all()
    if kind == "onehot":
        assert np.isfinite(out).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name: str) -> None:
    X, y = _toy_frame()
    adapter = make_adapter(name)
    Xa = adapter.fit_transform(X)
    params = {"lr": {}, "dt": {"max_depth": 3}, "rf": {"n_estimators": 20}}.get(name, {})
    model = make_model(name, params, seed=0, n_jobs=2, n_estimators=20)
    model = fit_model(model, name, Xa, y, cat_features=adapter.cat_cols)
    p = predict_proba(model, name, adapter.transform(X), adapter.cat_cols)
    assert p.shape == (len(y),)
    assert np.all((p >= 0) & (p <= 1))


def test_smotenc_imputation_is_persisted_and_integers_stay_integer() -> None:
    X, y = _toy_frame()
    res = apply_imbalance(X, y, "smotenc", seed=0)
    assert res.num_fill is not None and "speed_limit" in res.num_fill
    assert not res.X["speed_limit"].isna().any()
    assert np.allclose(res.X["n_veh"], np.round(res.X["n_veh"]))
    adapter = FeatureAdapter(kind="native", num_fill=res.num_fill).fit(res.X)
    out = adapter.transform(X)  # original data still has NaN speed limits
    assert not out["speed_limit"].isna().any()
    assert out["speed_limit"][X["speed_limit"].isna()].eq(res.num_fill["speed_limit"]).all()
