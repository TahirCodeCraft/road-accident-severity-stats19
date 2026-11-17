# Road-Collision Severity Prediction on STATS19

Code, data and results for the paper **"Feature Engineering, Imbalance Handling and
Calibration for Crash-Severity Prediction: An Out-of-Time Ablation on STATS19"**.

Tahir Ahmed Khan (25I-8025), Ahmed Qamar (25I-8049), Ali Raza (25I-8043)
MS Data Science, Data Science Tools and Techniques

- Paper (PDF): [`paper/main.pdf`](paper/main.pdf)
- Paper (Word): [`paper/main.docx`](paper/main.docx)
- LaTeX source: [`paper/`](paper/), see [`paper/README.md`](paper/README.md) for build steps
- Code notebook (executed): [`notebooks/Road_Accident_Severity_Walkthrough.ipynb`](notebooks/Road_Accident_Severity_Walkthrough.ipynb)

## Study in brief

We predict whether a police-reported injury collision in Great Britain is severe (fatal or
serious) or slight. Three techniques are switched on and off independently across six model
families (logistic regression, decision tree, random forest, XGBoost, LightGBM, CatBoost):

| Factor | Levels |
|---|---|
| Feature engineering (FE) | F0 raw features; F1 engineered features with training-only selection |
| Cost-sensitive learning (CS) | none; class weighting; random undersampling; SMOTE-NC |
| Calibration (CAL) | raw scores; Platt-style; isotonic; with five decision thresholds |

The data are split by time, and the 2025 test year was opened only once, after every model,
calibrator, threshold and contrast had been fixed and hashed in
[`experiments/frozen_manifest.json`](experiments/frozen_manifest.json).

| Split | Data | Collisions | Severe | Used for |
|---|---|---:|---:|---|
| FIT | 2023 | 104,258 | 23.9% | fitting models, encoders, feature selection, resampling |
| TUNE | 2024, odd ISO weeks | 51,241 | 24.7% | Optuna objective (PR-AUC), early stopping |
| CAL | 2024, even ISO weeks | 49,686 | 24.9% | calibrators, thresholds, model selection |
| TEST | 2025 | 101,525 | 26.2% | final evaluation only |

## Main results (2025 test year)

| | |
|---|---|
| Selected model (chosen on CAL) | LightGBM, engineered features, no reweighting, Platt calibration |
| PR-AUC / ROC-AUC | 0.496 (no-skill 0.262) / 0.730 |
| Brier score / ECE | 0.1675 / 0.009 |
| Threshold 1/6 (assumed 5:1 cost of a missed severe collision to a false alarm) | recall 0.856, precision 0.347, expected cost 1.072 → 0.612 (−43%) |

- **RQ1, feature engineering.** PR-AUC rose significantly for logistic regression, the decision
  tree, random forest and XGBoost (+0.004 to +0.017), but not for LightGBM or CatBoost.
- **RQ2, cost-sensitive learning.** Class weighting raised recall at the 0.5 threshold by 0.47 to
  0.52, but no significant F2 difference was found once each arm had its own tuned threshold.
  SMOTE-NC lowered PR-AUC in all six configurations where it was tested.
- **RQ3, calibration.** Calibration cut the ECE of weighted or undersampled models from about 0.2
  to about 0.01. For the final unweighted model, almost all of the cost reduction came from the
  threshold rather than from calibration.

Paired contrasts and subgroup results use seed 0; tables of means use three seeds. Details,
confidence intervals and limitations are in the paper.

## Repository layout

```
configs/              Hydra configuration (ablation matrix), feature lists, bank holidays
data/raw/             STATS19 2023–2025 CSVs and the DfT data guide (see data/README.md)
docs/                 implementation plan written before the experiments
experiments/cells/    one folder per ablation cell (48): config, Optuna trials, best parameters,
                      calibrators, frozen thresholds, CAL and TEST predictions for 3 seeds
experiments/frozen_manifest.json   selection, prespecified contrasts and artefact hashes
experiments/logs/     run logs
notebooks/            executed walkthrough notebook (reads saved outputs; no training)
paper/                IEEE LaTeX source, figures and compiled PDF
reports/figures/      EDA and result figures
reports/tables/       all result tables (CSV), including the leakage audit and statistics
reports/eda_findings.md   written EDA summary
scripts/              pipeline steps 01–14 (below)
src/                  package code: data/, features/, models/, training/, evaluation/, utils/
tests/                pytest suite (leakage guards, splits, encoders, calibration, statistics)
```

The 144 fitted models (about 1 GB) are not included. All their CAL and TEST predictions are, so
every table, figure and statistic can be regenerated without retraining.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) were used.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,report]"
pytest -q                      # 151 pass on a fresh clone; 6 more run once data/processed is built
```

## Reproducing the results

**From the saved outputs (minutes, no GPU).** These steps read `experiments/` and
`reports/tables/` only:

```bash
python scripts/11_report_assets.py          # result figures and tables in reports/
python scripts/14_paper_figures.py          # paper figures in paper/figures/
python scripts/13_make_notebook.py          # rebuild and execute the notebook
```

**Full pipeline (hours; XGBoost and CatBoost used a GPU).**

```bash
python scripts/01_build_dataset.py          # clean, leakage audit, collision-level table, splits
python scripts/02_eda.py                    # EDA on 2023–2024 only
python scripts/03_build_features.py         # F0 and F1full feature tables
python scripts/04_select_features.py        # training-only feature selection -> F1
python scripts/05_run_cells.py run.models=[lr,dt,rf,lgbm]
python scripts/05_run_cells.py run.models=[xgb,cat] run.device=cuda
python scripts/10_sensitivity.py build
python scripts/10_sensitivity.py run --model xgb --device cuda
python scripts/06_validate.py               # calibrators and thresholds on CAL
python scripts/07_freeze.py                 # fix selection and contrasts, hash artefacts
python scripts/08_evaluate_test.py          # one-time 2025 evaluation and paired statistics
python scripts/09_shap.py                   # TreeSHAP for the selected model
python scripts/10_sensitivity.py multiclass # three-class sensitivity analysis
```

`08_evaluate_test.py` refuses to run if any model, calibrator, feature table or config differs
from the hashes in the manifest. A full rerun therefore produces new models and must be frozen
again with `07_freeze.py` before the test year is scored. Experiments were logged to Weights &
Biases in offline mode; those local run files are not included.

## Data and licence

STATS19 road safety data are published by the UK Department for Transport under the Open
Government Licence v3.0. See [`data/README.md`](data/README.md) for the source and attribution.
