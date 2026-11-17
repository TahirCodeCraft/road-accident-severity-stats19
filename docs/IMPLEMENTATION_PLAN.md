# Implementation Plan — STATS19 Road Accident Severity (DSTT Project)

Team: Tahir Ahmed Khan (25I-8025), Ahmed Qamar (25I-8049), Ali Raza (25I-8043) — MS Data Science.
Target: collision-level binary Severe (collision_severity ∈ {1 Fatal, 2 Serious}) vs Slight (3).
Data: DfT STATS19 2023, 2024 (development) and 2025 (temporal test, untouched until freeze).

## 0. Data facts that drive the design (profiled 2023–2025)
- 104,258 / 100,927 / 101,525 collisions; `collision_index` unique; severe rate 23.9% / 24.8% / 26.2% (~3:1).
- `collision_injury_based=1` (CRaSH / injury-based recording) covers 54% / 59% / 87% of collisions;
  injury-based forces record more "serious" (28% vs 19% severe in 2023) → expect prior shift and
  calibration drift on 2025.
- Schema drift: `*_historic` cols and `special_conditions_at_site` ~1–3% missing in 2023, 59% 2024,
  87% 2025; `age_of_vehicle` 100% missing 2025; `driver_distance_banding`/`casualty_distance_banding`
  100% missing 2023; `local_authority_district` 100% missing; `junction_detail=99` 0/2/1427 rows.
- `did_police_officer_attend_scene_of_accident` strongly tied to severity (attendance follows severity).

## 1. Leakage audit (single source of truth: `src/data/schema.py`)
Prediction time (an assumed availability rule) = circumstances describable when the collision is
first reported (where/when/road/environment/who & what
was involved/pre-impact manoeuvres). Injury outcomes, impact consequences and police-processing
artefacts are excluded.

Collision table
- drop-ID: collision_index (join key), collision_year (split key), collision_ref_no.
- target: collision_severity.
- drop-leakage: enhanced_severity_collision, collision_adjusted_severity_serious,
  collision_adjusted_severity_slight (target-derived); number_of_casualties (injury-outcome count);
  did_police_officer_attend_scene_of_accident (decided by severity; kept OUTSIDE X as an evaluation
  stratifier).
- keep (borderline): collision_injury_based — reporting system of the force, fixed before the crash;
  measurement artefact → caveat in SHAP; removed in sensitivity S3.
- keep: number_of_vehicles, date→(month, day-of-year), time→hour, day_of_week, longitude, latitude,
  police_force, local_authority_ons_district (encoded), first_road_class, second_road_class, road_type,
  speed_limit, junction_detail, junction_control (42% structural missing → "not at junction"),
  pedestrian_crossing, carriageway_hazards, light_conditions, weather_conditions,
  road_surface_conditions, urban_or_rural_area, trunk_road_flag.
- drop-other: location_easting_osgr/northing (dup of lon/lat), local_authority_district (100% missing),
  local_authority_highway(_current) (redundant), lsoa_of_accident_location (>30k levels),
  first_road_number/second_road_number (identifiers), all *_historic columns,
  special_conditions_at_site (phased out).

Vehicle table (aggregated to collision level; order-free summaries only)
- drop-ID: collision_index, collision_year, collision_ref_no, vehicle_reference.
- keep: vehicle_type, towing_and_articulation, vehicle_manoeuvre, vehicle_direction_from/to,
  vehicle_location_restricted_lane, junction_location, vehicle_left_hand_drive,
  journey_purpose_of_driver, sex_of_driver, age_of_driver, engine_capacity_cc, propulsion_code,
  driver_imd_decile, escooter_flag.
- drop-other: age_band_of_driver (redundant), *_historic, age_of_vehicle (absent 2025),
  driver_distance_banding (absent 2023), generic_make_model, lsoa_of_driver.
- drop-leakage (Tier B, borderline — added back only in sensitivity S1): skidding_and_overturning,
  hit_object_in_carriageway, vehicle_leaving_carriageway, hit_object_off_carriageway,
  first_point_of_impact.

Casualty table (rows exist only for injured people → outcome-conditional)
- Only allowed signal: pedestrian-involved flag (any casualty_class == 3) and, for the first pedestrian
  (lowest casualty_reference): age_of_casualty, pedestrian_location, pedestrian_movement. Never counts.
- Everything else dropped (target-derived, injury composition, high cardinality, or absent in 2023).

Missing codes: -1 and data-guide "unknown (self reported)" codes (field-specific 9/99) → one missing
level; unseen categories → missing level. Row-count assert after every join.

## 2. Split design
| Split | Rows | Role |
|---|---|---|
| FIT  = 2023 | ~104k | fit models, encoders, selection, resampling |
| TUNE = 2024 odd ISO weeks | ~50k | early stopping, Optuna objective |
| CAL  = 2024 even ISO weeks | ~50k | calibrator fit, threshold choice, validation reporting, model selection |
| TEST = 2025 | ~101k | final evaluation only |

Refit: after HPO, refit each cell on FIT+TUNE with frozen hyperparameters, n_estimators =
round(1.2 × best_iter). CAL is never trained on → calibrator/threshold fitted on CAL predictions of the
same model that scores 2025. Validation numbers for calibrated variants via 5-fold cross-fitting in CAL.
Test guard: `load_split("test")` raises unless `allow_test=True` and `experiments/frozen_manifest.json`
exists.

## 3. Feature sets
F0 (raw): kept collision columns as-is (+ hour, month parsed); generic vehicle summaries (count per code
for each kept low-cardinality vehicle field, min/mean/max driver age and engine cc); pedestrian flag;
police_force categorical; ONS district frequency-encoded (FIT counts).

F1 (engineered) = F0 + temporal (hour/month sin-cos, is_weekend, is_night 22–05, rush_hour, weekend_night,
season, is_bank_holiday), environment (is_dark, dark_unlit, poor_weather, high_wind, wet_or_icy,
dark_and_wet, fine_but_wet, visibility_risk), road (speed numeric, high_speed ≥60, rural_high_speed,
rural_single_carriageway_60, is_junction, junction_complexity, uncontrolled_junction, is_roundabout,
is_motorway, road_class_pair, crossing_present, hazard_present), vehicle mix (single_vehicle, counts per
group car/van/HGV/bus/motorcycle/pedal cycle/e-scooter/other, has_vru, max/min mass class, mass_mismatch,
vru_vs_heavy, relative_heading for 2-vehicle crashes, any_overtaking/right_turn/stationary/reversing),
driver (age min/max/range, any_young <25, any_older 70+, n_age_unknown, share_male, any_work_journey,
any_electric, IMD min/mean), pedestrian (ped_child, ped_elderly, ped_crossing_group), interactions
(high_speed×has_vru, is_dark×ped_involved, rural×single_vehicle, speed_limit×single_vehicle), geography
(OOF target encoding m=50 for ONS district, police_force, 5 km grid; rotated coordinates).

Selection (FIT only, on F1): near-constant (>99.5%) → redundancy (|Spearman| or Cramér's V >0.95, keep
higher MI) → LightGBM permutation importance (PR-AUC drop), leave-one-quarter-out CV on 2023, keep
features with positive mean importance in ≥3/4 folds. Frozen list → `configs/features/f1_selected.yaml`.
Extra arm F1-full (no selection) for LR and LightGBM.

Adapters: LR one-hot (min_frequency=50) + median impute w/ indicators + scaling; DT/RF ordinal;
XGB (enable_categorical) / LGBM / CatBoost native categoricals + NaN.

## 4. Ablation matrix
Factors: FE ∈ {F0, F1}; CS ∈ {none, weight, RUS 1:1, SMOTE-NC 1:1}; CAL (post-hoc, free) ∈
{raw@0.5, raw@t*, platt@t*, isotonic@t*}.
Trained cells: 2 FE × 3 CS (none/weight/RUS) × 6 models = 36; + SMOTE-NC for LR/RF/LGBM × 2 FE = 6;
+ F1-full for LR/LGBM = 2 → 44 cells × 4 CAL levels.

Contrasts: RQ1 F1 vs F0 @ CS=none, raw. RQ2 each CS vs none on F0 and F1 (recall@0.5 AND F2/recall@t*,
PR-AUC — weighting mostly moves the operating point). RQ3 CAL levels on (F0,none) and (F1,best CS):
ΔBrier, ΔECE, cost (Platt is monotone → decisions equal raw@t*; calibration shows in Brier/ECE/
reliability and cost-based thresholds). Combined: (F1, best CS, calibrated@t*) vs (F0, none, raw@0.5).

HPO: Optuna TPE seed 42; 30 trials (15 for RF and SMOTE-NC cells) [amended from 40/20 for compute];
same space per model; objective PR-AUC on TUNE; boosters early-stop on TUNE logloss (patience 100);
resampling on training rows only. Refits: 3 seeds (mean ± sd) [amended from 5]; seed-0 model for paired
tests. Selection: highest cross-fitted CAL PR-AUC; ties within CI → lower Brier → simpler model.

Uncertainty on 2025: paired bootstrap (B=2000) 95% CIs for every Δ; McNemar at operating threshold;
DeLong for ROC-AUC; Holm across primary contrasts. Report val→test gap per technique; stratify by
collision_injury_based and police attendance.

Sensitivities (selected config): S1 add Tier-B crash mechanics; S2 missingness-blind; S3 drop
police_force, collision_injury_based and force TE; S4 three-class LightGBM (macro-F1).

## 5. Calibration and thresholds
Platt (logistic on logit) and isotonic fitted on CAL predictions; "CAL=on" uses lower cross-fitted Brier.
Primary t* = argmax F2 on CAL. Secondary: precision at recall ≥0.80; cost-based t = 1/6 (FN:FP = 5:1)
with expected cost. Metrics: Brier, ECE (15 equal-mass bins), reliability diagrams. Thresholds and
calibrators frozen to the manifest before TEST is loaded.

## 6. Compute placement
Local (16 CPU): data build, EDA, features, selection, LR/DT/RF/LightGBM HPO+refits, calibration, stats,
SHAP, sensitivities, report assets. GPU server (A5000, GPU 0, `/sdb-disk/administrator/experiments_2`):
XGBoost (`device=cuda`, xgboost 3.0.5) and CatBoost (`task_type=GPU`) HPO+refits, thread_count ≤ 4
(server CPU saturated). wandb in offline mode (no credentials configured).

## 7. Deliverables
IEEE-style report (abstract, intro/RQs, related work, data & leakage audit, methodology, setup, results
RQ1–RQ3 + combined + persistence + sensitivities, SHAP, discussion, conclusion, reproducibility),
tables/figures under `reports/`, updated defence slides.
