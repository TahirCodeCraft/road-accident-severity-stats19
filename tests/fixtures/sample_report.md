---
title: Predicting Road Accident Severity from STATS19 Records
subtitle: Feature engineering, cost-sensitive learning and calibration under a temporal holdout
authors:
  - Tahir Ahmed Khan (25I-8025)
  - Ahmed Qamar (25I-8049)
  - Ali Raza (25I-8043)
affiliation: Department of Data Science
course: DSTT Course Project
date: November 2026
---

<!-- SYNTHETIC TEST FIXTURE: all text and numbers are placeholders, not project results. -->


<!-- Sample source used by tests/test_docx_builder.py; it exercises every supported construct. -->

# Abstract

We predict whether a police-reported road collision in Great Britain is **fatal or serious
(KSI)** using the STATS19 collision and vehicle tables for 2021–2025. Models are trained on
2021–2023, calibrated and thresholded on 2024, and evaluated *once* on the untouched 2025 holdout.
Engineered features raise PR-AUC from 0.312 to 0.348, while isotonic calibration reduces the Brier
score by 9 % without changing ranking quality [1], [3]–[5].

> Key finding: cost-sensitive gradient boosting with isotonic calibration achieves the best
> expected cost on 2025 data, and its operating threshold frozen on 2024 transfers without drift.

# 1 Introduction

Road collisions remain a leading cause of preventable injury. Severity models support the triage of
police resources and the prioritisation of engineering interventions [2]. This report asks three
research questions:

1. Does domain-driven feature engineering improve discrimination over raw coded fields?
2. Does cost-sensitive learning beat post-hoc threshold moving?
   - measured by expected cost at a 10:1 false-negative to false-positive cost ratio;
   - and by recall at fixed precision.
3. Are calibrated probabilities stable across the 2024 → 2025 shift?

## 1.1 Contributions

- A leakage-audited feature set built with `polars` from the collision and vehicle tables.
- A full ablation grid over features, imbalance handling and calibration:
  - three feature sets (F0 raw, F1 engineered, F2 engineered + target encoding);
  - four learners (logistic regression, random forest, LightGBM, XGBoost).
- A strictly temporal protocol: no random splits anywhere in the pipeline.

# 2 Data and Methods

## 2.1 Dataset

Table: Temporal split of the STATS19 collisions used in this study

| Split | Years | Collisions | KSI rate (%) | Role |
|:------|:-----:|-----------:|-------------:|:-----|
| Train | 2021–2023 | 312,480 | 22.4 | Model fitting and Optuna tuning |
| CAL | 2024 | 98,115 | 23.1 | Calibration and threshold selection |
| Test | 2025 | 95,902 | 23.6 | Single final evaluation |

## 2.2 Feature engineering

Features fall into four families: temporal (hour, weekday, bank holiday), road environment (speed
limit, junction detail, light and weather), vehicle mix (counts of motorcycles, HGVs and pedal
cycles) and location (urban/rural indicator, police force). Missing codes such as `-1` and `9` are
mapped to an explicit *unknown* level rather than imputed. The Brier score is computed as
Brier = mean((p − y)^2) over all test collisions.

### 2.2.1 Selection

Features with a mutual information below 0.001 or a variance-inflation factor above 10 are dropped,
leaving **41** of 63 candidates.

\pagebreak

# 3 Results

![Precision–recall curves on the 2025 holdout for the four learners](figures/fig_sample.png){width=12cm}

Table: Test-set metrics (2025) for the selected configuration of each learner

| Model | PR-AUC | ROC-AUC | Brier | ECE | Recall@t* |
|:------|-------:|--------:|------:|----:|----------:|
| Logistic regression | 0.301 | 0.702 | 0.162 | 0.011 | 0.71 |
| Random forest | 0.329 | 0.721 | 0.158 | 0.019 | 0.74 |
| LightGBM (cost-sensitive) | **0.348** | **0.736** | **0.154** | **0.008** | **0.78** |
| XGBoost | 0.344 | 0.733 | 0.155 | 0.009 | 0.77 |

Gradient-boosted trees dominate the linear baseline on every metric. The gap between LightGBM and
XGBoost is within the seed-to-seed standard deviation (± 0.004 PR-AUC).

![Reliability diagram of the selected model before and after isotonic calibration](figures/fig_sample.png)

# 4 Conclusion

Engineered features and cost-sensitive learning each contribute measurable gains, and calibration
fitted on 2024 transfers to 2025 without degradation.

# References

[1] K. G. Le, P. Liu, and L.-T. Lin, "Determining the road traffic accident hotspots using GIS-based
temporal-spatial statistical analytic techniques in Hanoi, Vietnam," *Geo-spatial Information
Science*, vol. 23, no. 2, pp. 153–164, 2020.
[2] Department for Transport, "Reported road casualties Great Britain: notes, definitions and
guidance (STATS19)," UK Government, 2024.
[3] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. KDD*, 2016,
pp. 785–794.
[4] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Proc.
NeurIPS*, 2017.
[5] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning,"
in *Proc. ICML*, 2005, pp. 625–632.
