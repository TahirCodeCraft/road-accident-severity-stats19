"""Tests for src.models.calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import brier_score
from src.models.calibration import (
    CalibrationMethod,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    cross_fit_calibrated,
    load_calibrator,
    make_calibrator,
    select_calibrator,
)


def _miscalibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """True P(y=1) = sigmoid(z); reported probability is over-confident and shifted."""
    rng = np.random.default_rng(seed)
    z = rng.normal(-1.2, 1.0, size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    p_raw = 1.0 / (1.0 + np.exp(-(2.5 * z + 1.0)))
    return y, p_raw


@pytest.mark.parametrize("cls", [PlattCalibrator, IsotonicCalibrator])
def test_calibrators_reduce_brier_out_of_sample(cls: type) -> None:
    y_fit, p_fit = _miscalibrated(20_000, seed=0)
    y_eval, p_eval = _miscalibrated(20_000, seed=1)
    cal = cls().fit(y_fit, p_fit)
    p_cal = cal.predict(p_eval)
    assert p_cal.shape == p_eval.shape
    assert np.all((p_cal >= 0) & (p_cal <= 1))
    assert brier_score(y_eval, p_cal) < brier_score(y_eval, p_eval) - 0.01


def test_platt_recovers_known_map() -> None:
    y, p = _miscalibrated(100_000, seed=2)
    slope, intercept = PlattCalibrator().fit(y, p).coef
    assert slope == pytest.approx(1 / 2.5, abs=0.03)
    assert intercept == pytest.approx(-1.0 / 2.5, abs=0.05)


def test_isotonic_clips_out_of_bounds_and_is_monotone() -> None:
    y, p = _miscalibrated(5000, seed=3)
    cal = IsotonicCalibrator().fit(y, np.clip(p, 0.1, 0.9))
    grid = np.linspace(0, 1, 101)
    out = cal.predict(grid)
    assert np.all(np.diff(out) >= 0)
    assert np.all(np.isfinite(out))


def test_identity_calibrator() -> None:
    p = np.array([0.0, 0.3, 1.0])
    assert np.array_equal(IdentityCalibrator().fit([0, 1, 1], p).predict(p), p)


def test_unfitted_predict_raises() -> None:
    with pytest.raises(RuntimeError):
        PlattCalibrator().predict([0.5])
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict([0.5])


@pytest.mark.parametrize("method", list(CalibrationMethod))
def test_save_load_roundtrip(tmp_path: Path, method: CalibrationMethod) -> None:
    y, p = _miscalibrated(2000, seed=4)
    cal = make_calibrator(method).fit(y, p)
    path = tmp_path / "sub" / f"{method.value}.joblib"
    cal.save(path)
    loaded = load_calibrator(path)
    assert loaded.method is method
    np.testing.assert_allclose(loaded.predict(p), cal.predict(p))


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_cross_fit_returns_oof_probs(method: str) -> None:
    y, p = _miscalibrated(10_000, seed=5)
    oof = cross_fit_calibrated(y, p, method, n_splits=5, seed=0)
    assert oof.shape == p.shape
    assert np.all((oof >= 0) & (oof <= 1))
    assert brier_score(y, oof) < brier_score(y, p)
    in_sample = make_calibrator(method).fit(y, p).predict(p)
    assert not np.allclose(oof, in_sample)
    np.testing.assert_array_equal(oof, cross_fit_calibrated(y, p, method, n_splits=5, seed=0))


def test_select_calibrator_prefers_platt_for_sigmoid_distortion() -> None:
    y, p = _miscalibrated(5000, seed=6)
    sel = select_calibrator(y, p, n_splits=5, seed=0)
    assert set(sel.cv_brier) == {"platt", "isotonic"}
    assert sel.cv_brier[sel.method.value] == min(sel.cv_brier.values())
    assert sel.method is CalibrationMethod.PLATT
    assert sel.oof_prob.shape == p.shape
    assert sel.calibrator.predict(p).shape == p.shape


def test_select_calibrator_prefers_isotonic_for_non_sigmoid_distortion() -> None:
    rng = np.random.default_rng(8)
    true_p = rng.uniform(0, 1, size=40_000)
    y = (rng.random(true_p.size) < true_p).astype(int)
    p_raw = np.where(true_p < 0.5, true_p * 0.2, 0.9 + true_p * 0.1)  # step-like, non-logistic
    sel = select_calibrator(y, p_raw, seed=0)
    assert sel.method is CalibrationMethod.ISOTONIC
