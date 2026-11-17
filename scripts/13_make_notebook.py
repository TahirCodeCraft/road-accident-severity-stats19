"""Generate and execute the walkthrough notebook (reads saved artefacts only; no training).

Output: notebooks/Road_Accident_Severity_Walkthrough.ipynb (executed, outputs embedded).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import nbformat  # noqa: E402
from nbconvert.preprocessors import ExecutePreprocessor  # noqa: E402

from src.paths import PROJECT_ROOT  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)
NB_PATH = PROJECT_ROOT / "notebooks" / "Road_Accident_Severity_Walkthrough.ipynb"

CELLS: list[tuple[str, str]] = [
    ("md", """# Road Accident Severity Prediction — project walkthrough
**DSTT project · Tahir Ahmed Khan (25I-8025), Ahmed Qamar (25I-8049), Ali Raza (25I-8043)**

This notebook walks through the finished pipeline using the saved artefacts. The heavy steps
(training 144 models, GPU runs) are reproducible scripts in `scripts/` (see README); this notebook
only loads their outputs, so it runs in seconds."""),
    ("code", """import json, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
T, F = ROOT / "reports" / "tables", ROOT / "reports" / "figures"
pd.set_option("display.max_columns", 30, "display.width", 160, "display.precision", 4)"""),
    ("md", "## 1. Data and splits\nUK DfT STATS19 2023–2025, one row per collision. "
           "Target: Severe (fatal/serious) vs Slight."),
    ("code", """display(pd.read_csv(T / "split_summary.csv"))
display(pd.read_csv(T / "data_quality.csv").head(8))"""),
    ("md", "## 2. Leakage audit\nEvery raw column has exactly one decision "
           "(`src/data/schema.py`); tests enforce it."),
    ("code", """audit = pd.read_csv(T / "leakage_audit.csv")
display(audit["decision"].value_counts().to_frame())
mask = audit["decision"].str.contains("leak|tier", regex=True)
display(audit[mask][["table", "name", "decision", "reason"]])"""),
    ("md", "## 3. Exploratory analysis (2023–2024 only)"),
    ("code", """display(Markdown((ROOT / "reports" / "eda_findings.md").read_text()))
for f in ["eda_02_monthly_trend_reporting.png", "eda_03_hour_dow_heatmap.png",
          "eda_04_road_factors.png", "eda_06_vehicle_mix.png"]:
    display(Image(filename=str(F / f), width=850))"""),
    ("md", "## 4. Features and selection"),
    ("code", """sel = pd.read_csv(T / "feature_selection.csv")
display(sel["selected"].value_counts().rename("count").to_frame())
display(sel[sel["selected"]].sort_values("mean_importance", ascending=False)
        [["feature", "kind", "mean_importance", "n_positive_folds"]].head(20))"""),
    ("md", "## 5. Pipeline and evaluation design"),
    ("code", """display(Image(filename=str(F / "fig_01_pipeline.png"), width=950))
manifest = json.loads((ROOT / "experiments" / "frozen_manifest.json").read_text())
print("Frozen at:", manifest["frozen_at"])
print("Selected model:", manifest["selected"])
print("CAL-best imbalance strategy per (features, model):", manifest["best_cs"])"""),
    ("md", "## 6. Results on the unseen 2025 test year\n### Best configuration per model family"),
    ("code", """display(pd.read_csv(T / "main_results.csv"))
display(Image(filename=str(F / "fig_02_ablation_prauc.png"), width=900))
display(Image(filename=str(F / "fig_14_pr_roc.png"), width=950))"""),
    ("md", "### Research questions — prespecified primary contrasts (Holm-adjusted)"),
    ("code", """display(pd.read_csv(T / "rq_contrasts_primary.csv"))
for f in ["fig_04_rq1_prauc.png", "fig_05_rq2_recall05.png", "fig_06_rq2_f2.png",
          "fig_07_rq3_brier.png", "fig_08_rq3_cost.png", "fig_09_combined_f2.png"]:
    if (F / f).exists():
        display(Image(filename=str(F / f), width=900))"""),
    ("md", "### Calibration, thresholds and errors of the selected model"),
    ("code", """for f in ["fig_11_reliability_selected.png", "fig_12_threshold_curves.png",
          "fig_13_confusion.png", "fig_15_val_test_prauc.png", "fig_16_stratified.png"]:
    if (F / f).exists():
        display(Image(filename=str(F / f), width=800))"""),
    ("md", "## 7. Explainability (SHAP) and sensitivity analyses"),
    ("code", """display(pd.read_csv(T / "shap_importance.csv").head(15))
print(json.loads((T / "shap_stability.json").read_text()))
for f in ["shap_bar_test.png", "shap_beeswarm_test.png", "fig_10_sensitivity.png"]:
    if (F / f).exists():
        display(Image(filename=str(F / f), width=800))
for f in ["sensitivity.csv", "sensitivity_s4_multiclass.csv"]:
    if (T / f).exists():
        display(pd.read_csv(T / f))"""),
    ("md", "## 8. Tests\nThe full pytest suite is run as part of the build "
           "(`pytest -q`); the summary is stored in `reports/tables/test_summary.txt`."),
    ("code", """p = T / "test_summary.txt"
print(p.read_text() if p.exists() else "run: pytest -q")"""),
]


def build_notebook() -> nbformat.NotebookNode:
    """Assemble the notebook from :data:`CELLS`.

    Returns:
        Notebook object.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    for kind, src in CELLS:
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md"
                        else nbformat.v4.new_code_cell(src))
    return nb


def main() -> None:
    """Entry point."""
    setup_logging()
    nb = build_notebook()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(NB_PATH.parent)}})
    nbformat.write(nb, NB_PATH)
    logger.info("Executed notebook written: %s", NB_PATH)


if __name__ == "__main__":
    main()
