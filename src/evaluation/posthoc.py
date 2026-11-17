"""Post-hoc calibration, threshold selection (on CAL) and frozen evaluation (on TEST).

The CAL factor of the ablation is applied here without retraining:

* variants: ``raw`` (uncalibrated), ``platt``, ``isotonic`` — calibrators fitted on CAL predictions
  of the refit model; validation metrics for calibrated variants use 5-fold cross-fitted (OOF)
  probabilities so CAL numbers are honest.
* operating points (thresholds chosen on CAL, frozen before TEST):
  ``t050`` fixed 0.5; ``tf2`` max F2; ``trec80`` best precision s.t. recall >= 0.80;
  ``tcost`` Bayes threshold 1/(1+5) for FN:FP = 5:1 (valid only for calibrated probabilities);
  ``tcostemp`` empirical cost-minimising threshold on CAL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import binary_metrics
from src.evaluation.thresholds import ThresholdConfig, ThresholdPolicy, choose_threshold
from src.models.calibration import (
    Calibrator,
    IdentityCalibrator,
    cross_fit_calibrated,
    load_calibrator,
    make_calibrator,
)
from src.paths import FROZEN_MANIFEST_PATH

logger = logging.getLogger(__name__)

VARIANTS = ("raw", "platt", "isotonic")
FN_COST = 5.0
FP_COST = 1.0
OPERATING_POINTS: dict[str, ThresholdConfig] = {
    "t050": ThresholdConfig(policy=ThresholdPolicy.FIXED, fixed=0.5),
    "tf2": ThresholdConfig(policy=ThresholdPolicy.MAX_FBETA, beta=2.0),
    "trec80": ThresholdConfig(policy=ThresholdPolicy.PRECISION_AT_RECALL, min_recall=0.80),
    "tcost": ThresholdConfig(policy=ThresholdPolicy.COST_BAYES, fn_cost=FN_COST, fp_cost=FP_COST),
    "tcostemp": ThresholdConfig(policy=ThresholdPolicy.COST_EMPIRICAL, fn_cost=FN_COST,
                                fp_cost=FP_COST),
}
METRIC_KEYS = ("precision", "recall", "f1", "f2", "specificity", "balanced_accuracy",
               "pred_pos_rate", "pr_auc", "roc_auc", "brier", "log_loss", "ece", "expected_cost",
               "tp", "fp", "tn", "fn", "prevalence", "n")


@dataclass
class FrozenDecision:
    """Everything decided on CAL for one (cell, seed).

    Attributes:
        thresholds: ``{variant: {operating_point: threshold}}``.
        best_calibrator: Calibrated variant with the lower cross-fitted CAL Brier.
        cv_brier: Cross-fitted CAL Brier per variant.
    """

    thresholds: dict[str, dict[str, float]]
    best_calibrator: str
    cv_brier: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        """Serialisable representation."""
        return {"thresholds": self.thresholds, "best_calibrator": self.best_calibrator,
                "cv_brier": self.cv_brier}


def metric_row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute the reported metric set at a threshold.

    Args:
        y: Labels.
        p: Probabilities.
        threshold: Decision threshold.

    Returns:
        Flat metric dict.
    """
    m = binary_metrics(y, p, threshold=threshold, fn_cost=FN_COST, fp_cost=FP_COST).to_dict()
    return {k: m[k] for k in METRIC_KEYS} | {"threshold": float(threshold)}


def fit_calibrators(y_cal: np.ndarray, p_cal: np.ndarray) -> dict[str, Calibrator]:
    """Fit every calibration variant on all CAL rows.

    Args:
        y_cal: CAL labels.
        p_cal: Raw CAL probabilities.

    Returns:
        ``{variant: fitted calibrator}``.
    """
    cals: dict[str, Calibrator] = {"raw": IdentityCalibrator().fit(y_cal, p_cal)}
    for v in ("platt", "isotonic"):
        cals[v] = make_calibrator(v).fit(y_cal, p_cal)
    return cals


def validate_on_cal(
    y_cal: np.ndarray, p_cal: np.ndarray, seed: int = 42
) -> tuple[FrozenDecision, dict[str, Calibrator], list[dict[str, Any]]]:
    """Choose calibrators and thresholds on CAL and report CAL validation metrics.

    Args:
        y_cal: CAL labels.
        p_cal: Raw CAL probabilities of the refit model.
        seed: Cross-fitting seed.

    Returns:
        ``(frozen decision, calibrators fitted on all CAL, validation metric rows)``.
    """
    # Reporting probabilities: cross-fitted (out-of-fold) calibration so CAL scores are honest.
    oof = {"raw": p_cal}
    for v in ("platt", "isotonic"):
        oof[v] = cross_fit_calibrated(y_cal, p_cal, v, n_splits=5, seed=seed)
    cv_brier = {v: float(np.mean((oof[v] - y_cal) ** 2)) for v in VARIANTS}
    best = min(("platt", "isotonic"), key=lambda v: cv_brier[v])
    # Deployment: calibrators fitted on all CAL; thresholds chosen on *their* outputs so the frozen
    # threshold lives on the same probability scale that is applied to TEST.
    calibrators = fit_calibrators(y_cal, p_cal)
    deploy = {v: np.clip(calibrators[v].predict(p_cal), 0.0, 1.0) for v in VARIANTS}
    thresholds: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    for v in VARIANTS:
        thresholds[v] = {}
        for op, cfg in OPERATING_POINTS.items():
            thresholds[v][op] = float(choose_threshold(y_cal, deploy[v], cfg).threshold)
            # CAL operating metrics: OOF probabilities with a threshold re-chosen on them
            # (in-sample threshold -> optimistic; TEST is the honest estimate).
            t_rep = float(choose_threshold(y_cal, oof[v], cfg).threshold)
            rows.append({"variant": v, "op": op} | metric_row(y_cal, oof[v], t_rep))
    return FrozenDecision(thresholds, best, cv_brier), calibrators, rows


def evaluate_frozen(
    y: np.ndarray, p_raw: np.ndarray, decision: FrozenDecision, calibrators: dict[str, Calibrator]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Apply frozen calibrators and thresholds to new data (TEST).

    Args:
        y: Labels.
        p_raw: Raw model probabilities.
        decision: Frozen CAL decision.
        calibrators: Calibrators fitted on CAL.

    Returns:
        ``(metric rows, {variant: probabilities})``.
    """
    probs = {v: np.clip(calibrators[v].predict(p_raw), 0.0, 1.0) for v in VARIANTS}
    rows = [
        {"variant": v, "op": op} | metric_row(y, probs[v], decision.thresholds[v][op])
        for v in VARIANTS for op in OPERATING_POINTS
    ]
    return rows, probs


def save_decision(out_dir: Path, seed: int, decision: FrozenDecision,
                  calibrators: dict[str, Calibrator]) -> None:
    """Persist the frozen decision and calibrators of one (cell, seed).

    Args:
        out_dir: Cell directory.
        seed: Refit seed.
        decision: Frozen decision.
        calibrators: Fitted calibrators.
    """
    (out_dir / f"frozen_seed{seed}.json").write_text(json.dumps(decision.to_json(), indent=2))
    for v in ("platt", "isotonic"):
        calibrators[v].save(out_dir / f"calibrator_{v}_seed{seed}.joblib")


def load_decision(out_dir: Path, seed: int) -> tuple[FrozenDecision, dict[str, Calibrator]]:
    """Load a frozen decision and its calibrators.

    Args:
        out_dir: Cell directory.
        seed: Refit seed.

    Returns:
        ``(decision, calibrators)``.
    """
    d = json.loads((out_dir / f"frozen_seed{seed}.json").read_text())
    cals: dict[str, Calibrator] = {"raw": IdentityCalibrator()}
    for v in ("platt", "isotonic"):
        cals[v] = load_calibrator(out_dir / f"calibrator_{v}_seed{seed}.joblib")
    return FrozenDecision(d["thresholds"], d["best_calibrator"], d["cv_brier"]), cals


def require_frozen_manifest() -> dict[str, Any]:
    """Return the frozen manifest or raise if the experiment has not been frozen.

    Returns:
        Parsed manifest.

    Raises:
        PermissionError: If the manifest does not exist.
    """
    if not FROZEN_MANIFEST_PATH.exists():
        raise PermissionError("TEST is sealed: run scripts/07_freeze.py first.")
    return json.loads(FROZEN_MANIFEST_PATH.read_text())
