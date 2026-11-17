"""Freeze all CAL-based decisions before the 2025 TEST year is opened.

Selection rules (decided on CAL only):
* best CS per (model, feature set): highest mean CAL F2 at the F2-optimal threshold of the best
  calibrator (across seeds);
* final model: highest mean CAL PR-AUC (raw scores) over the ablation cells; ties within 0.002 go to
  lower Brier of the best calibrator, then to the simpler model.

Writes ``experiments/frozen_manifest.json`` with config/artefact hashes, which unlocks TEST.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from pathlib import Path as _Path

import pandas as pd

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import CONFIGS_DIR, EXPERIMENTS_DIR, FROZEN_MANIFEST_PATH, TABLES_DIR
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)
SIMPLICITY = {"lr": 0, "dt": 1, "rf": 2, "lgbm": 3, "xgb": 4, "cat": 5}
PR_AUC_TIE = 0.002


def file_hash(path: Path) -> str:
    """SHA-256 of a file (first 16 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def best_cs(val: pd.DataFrame) -> dict[str, str]:
    """Pick the best imbalance strategy per (model, feature set) on CAL.

    Args:
        val: Validation table.

    Returns:
        ``{"<feature_set>__<model>": cs}``.
    """
    tagless = val[(val["tag"].fillna("") == "") & (val["op"] == "tf2")]
    tagless = tagless[tagless["variant"] == tagless["best_calibrator"]]
    agg = tagless.groupby(["feature_set", "model", "cs"])["f2"].mean().reset_index()
    out = {}
    for (fs, m), g in agg.groupby(["feature_set", "model"]):
        out[f"{fs}__{m}"] = str(g.sort_values("f2", ascending=False).iloc[0]["cs"])
    return out


def select_final(val: pd.DataFrame) -> dict:
    """Select the final model configuration on CAL.

    Args:
        val: Validation table.

    Returns:
        Dict describing the selected cell.
    """
    v = val[(val["tag"].fillna("") == "") & (val["op"] == "t050")]
    raw = v[v["variant"] == "raw"].groupby("cell_id")["pr_auc"].mean()
    cal = v[v["variant"] == v["best_calibrator"]].groupby("cell_id")["brier"].mean()
    model = v.groupby("cell_id")["model"].first()
    cand = pd.DataFrame({"pr_auc": raw, "brier": cal, "model": model})
    top = cand["pr_auc"].max()
    tied = cand[cand["pr_auc"] >= top - PR_AUC_TIE].copy()
    tied["simplicity"] = tied["model"].map(SIMPLICITY)
    chosen = tied.sort_values(["brier", "simplicity"]).index[0]
    logger.info("Top CAL PR-AUC %.4f; %d cells within tie margin; selected %s", top, len(tied),
                chosen)
    return {"cell_id": chosen, "cal_pr_auc": float(cand.loc[chosen, "pr_auc"]),
            "cal_brier_calibrated": float(cand.loc[chosen, "brier"]),
            "tied_cells": tied.index.tolist()}


SENSITIVITY_CELLS = ("F1S1__none__xgb__S1", "F1S2__none__xgb__S2", "F1__none__xgb__S3",
                     "F1__none__xgb__S5")
SEEDS = (0, 1, 2)


def expected_cells() -> list[str]:
    """Cell ids the experiment design requires (ablation matrix + sensitivity analyses).

    Returns:
        Sorted cell ids.
    """
    from omegaconf import OmegaConf

    exp = OmegaConf.load(CONFIGS_DIR / "experiment" / "ablation.yaml")
    cells = [f"{fs}__{cs}__{m}" for fs in exp.feature_sets for cs in exp.cs_strategies
             for m in exp.models]
    cells += [f"{fs}__smotenc__{m}" for fs in exp.feature_sets for m in exp.smotenc_models]
    cells += [f"F1full__none__{m}" for m in exp.f1full_models]
    return sorted(cells + list(SENSITIVITY_CELLS))


def check_complete(cells_dir: Path, val: pd.DataFrame) -> list[str]:
    """List problems that make the experiment unfit to freeze.

    Args:
        cells_dir: Directory with one folder per cell.
        val: Validation table.

    Returns:
        Human-readable problems (empty if complete).
    """
    problems = []
    validated = set(val["cell_id"])
    for cid in expected_cells():
        d = cells_dir / cid
        if not (d / "summary.json").exists():
            problems.append(f"{cid}: not trained")
            continue
        for s in SEEDS:
            for f in (f"model_seed{s}.joblib", f"preds_cal_seed{s}.parquet",
                      f"frozen_seed{s}.json"):
                if not (d / f).exists():
                    problems.append(f"{cid}: missing {f}")
        if cid not in validated:
            problems.append(f"{cid}: not in validation table")
    return problems


def provenance_hashes() -> dict[str, dict[str, str]]:
    """Hashes of every artefact that a TEST evaluation depends on.

    Returns:
        ``{"artefacts": ..., "features": ..., "source": ..., "configs": ...}``.
    """
    from src.paths import PROCESSED_DIR, PROJECT_ROOT

    cells_dir = EXPERIMENTS_DIR / "cells"
    artefacts = {
        p.relative_to(EXPERIMENTS_DIR).as_posix(): file_hash(p)
        for p in sorted(cells_dir.glob("*/*"))
        if p.is_file() and not p.name.startswith("preds_test")
    }
    features = {p.name: file_hash(p)
                for p in sorted((PROCESSED_DIR / "features").glob("*_refit.parquet"))}
    source = {p.relative_to(PROJECT_ROOT).as_posix(): file_hash(p)
              for p in sorted([*(PROJECT_ROOT / "src").rglob("*.py"),
                               *(PROJECT_ROOT / "scripts").glob("*.py")])}
    configs = {p.relative_to(CONFIGS_DIR).as_posix(): file_hash(p)
               for p in sorted(CONFIGS_DIR.rglob("*.yaml"))}
    return {"artefacts": artefacts, "features": features, "source": source, "configs": configs}


def frozen_contrasts(cells: list[str], best: dict[str, str]) -> list[str]:
    """Prespecify the TEST contrast family (keys checked again by scripts/08).

    Args:
        cells: Frozen cell ids.
        best: CAL-selected CS per ``<feature_set>__<model>``.

    Returns:
        Contrast keys.
    """
    import importlib.util
    import sys as _sys

    path = Path(__file__).with_name("08_evaluate_test.py")
    spec = importlib.util.spec_from_file_location("evaluate_test", path)
    module = importlib.util.module_from_spec(spec)
    _sys.modules["evaluate_test"] = module
    spec.loader.exec_module(module)
    models = sorted({c.split("__")[2] for c in cells})
    return module.contrast_keys(module.build_contrasts(set(cells), best, models))


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an existing manifest")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    setup_logging()
    if FROZEN_MANIFEST_PATH.exists() and not args.force:
        raise SystemExit("Manifest already exists; TEST may have been opened. Use --force only "
                         "if TEST has not been evaluated.")
    val = pd.read_csv(TABLES_DIR / "validation_results.csv")
    problems = check_complete(EXPERIMENTS_DIR / "cells", val)
    if problems and not args.allow_incomplete:
        raise SystemExit("Cannot freeze, experiment incomplete:\n" + "\n".join(problems))
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": f"max mean CAL PR-AUC (raw); ties within {PR_AUC_TIE} -> lower "
                          "calibrated Brier -> simpler model",
        "selected": select_final(val),
        "best_cs": best_cs(val),
        "cells": expected_cells(),
        "seeds": list(SEEDS),
        "incomplete": problems,
        "hashes": provenance_hashes(),
    }
    manifest["contrasts"] = frozen_contrasts(manifest["cells"], manifest["best_cs"])
    FROZEN_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    logger.info("Frozen manifest written: %s (%d cells)", FROZEN_MANIFEST_PATH,
                len(manifest["cells"]))


if __name__ == "__main__":
    main()
