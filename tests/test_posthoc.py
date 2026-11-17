"""Tests for post-hoc calibration/threshold freezing and statistical edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.posthoc import evaluate_frozen, load_decision, save_decision, validate_on_cal
from src.evaluation.stats import delong_test
from src.evaluation.thresholds import min_cost_threshold


def _synthetic(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 1.2)))).astype(int)
    return y, 1 / (1 + np.exp(-2 * z))  # over-confident, shifted scores


def test_frozen_thresholds_live_on_deployed_calibrator_scale(tmp_path: Path) -> None:
    y, p = _synthetic()
    decision, cals, _ = validate_on_cal(y, p)
    from src.evaluation.thresholds import ThresholdConfig, choose_threshold
    for v in ("platt", "isotonic"):
        q = np.clip(cals[v].predict(p), 0, 1)
        t = choose_threshold(y, q, ThresholdConfig()).threshold
        assert decision.thresholds[v]["tf2"] == t


def test_platt_threshold_decisions_match_raw_on_cal() -> None:
    y, p = _synthetic()
    decision, cals, _ = validate_on_cal(y, p)
    raw_dec = p >= decision.thresholds["raw"]["tf2"]
    platt_dec = np.clip(cals["platt"].predict(p), 0, 1) >= decision.thresholds["platt"]["tf2"]
    assert (raw_dec == platt_dec).mean() > 0.999  # Platt is monotone


def test_decision_roundtrip_and_calibration_improves_test_brier(tmp_path: Path) -> None:
    y, p = _synthetic(12000)
    decision, cals, _ = validate_on_cal(y[:6000], p[:6000])
    save_decision(tmp_path, 0, decision, cals)
    d2, c2 = load_decision(tmp_path, 0)
    assert d2.thresholds == decision.thresholds
    rows, _ = evaluate_frozen(y[6000:], p[6000:], d2, c2)
    brier = {r["variant"]: r["brier"] for r in rows if r["op"] == "t050"}
    assert brier["platt"] < brier["raw"] and brier["isotonic"] < brier["raw"]


def test_min_cost_threshold_can_predict_nobody() -> None:
    y = np.array([1] + [0] * 9)
    p = np.full(10, 0.4)
    t = min_cost_threshold(y, p, fn_cost=5.0, fp_cost=1.0)
    assert t > 0.4  # all-negative (cost 5) beats all-positive (cost 9)


def test_delong_degenerate_separation() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    res = delong_test(y, p, 1 - p)
    assert res.delta == 1.0 and res.p_value == 0.0
    assert delong_test(y, p, p).p_value == 1.0
