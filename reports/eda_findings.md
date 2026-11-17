# Exploratory data analysis: key findings

Scope: STATS19 collisions in Great Britain. All severity (label) statistics use the 2023–2024
development years only (n = 205,185). The 2025 test year (n = 101,525) appears only in label-free
covariate views: reporting-system share and missingness. Rates are given with 95% Wilson
confidence intervals. The full table is in `reports/tables/eda_summary.csv`, and the scripts that
produce it are `scripts/02_eda.py` and `src/evaluation/eda.py`.

1. **Class balance.** The positive class (Fatal or Serious) makes up about one quarter of
   collisions: 23.9% [23.7, 24.2] in 2023 and 24.8% [24.6, 25.1] in 2024. Fatal collisions are
   rare (1.5% in both years), so the binary target is driven almost entirely by "Serious".
   The imbalance is moderate, at roughly 3:1
   (`figures/eda_01_class_balance.png`).

2. **Reporting-system drift.** The share of collisions recorded by injury-based systems
   (CRaSH/COPA) rose from 53.7% in 2023 to 58.8% in 2024 and then jumped to 86.5% in 2025.
   The monthly severe rate moved between about 22% and 28% over 2023–2024 and peaked in
   August 2024 (`figures/eda_02_monthly_trend_reporting.png`). Because the 2025 covariate mix
   lies far outside the development range, a shift in the prior and in calibration is expected
   on the test year.

3. **Recording system as a measurement effect.** Injury-based recording gives a much higher
   severe rate than legacy severity-based recording: 28.3% [28.0, 28.5] vs 19.4% [19.2, 19.7]
   over 2023–2024. Within each system the rate rose by only 0.5 percentage points from 2023 to
   2024 (28.0→28.5% and 19.2→19.7%). The overall rate rose by 0.9 points, so about half of the
   year-on-year increase reflects the change in system mix rather than in road risk
   (`figures/eda_09_reporting_system.png`).

4. **Time of day and week.** Severity is highest at night. The rate is about 29–33% between
   22:00 and 05:59 and lowest in the 08:00 morning peak (19.2% [18.6, 19.9]). Weekends are more
   severe than weekdays: Sunday 27.6% and Saturday 26.7%, against 22.8–24.0% from Monday to
   Friday. Across the hour × day cells, the rate ranges from 18% (Tuesday 08:00) to 39%
   (Thursday 05:00) (`figures/eda_03_hour_dow_heatmap.png`).

5. **Speed limit and rurality.** The severe rate rises steadily with the speed limit, from
   19.7% at 20 mph to 34.3% at 60 mph. It then falls back to 24.9% on 70 mph roads, which are
   mostly motorways and dual carriageways with separated traffic. Rural collisions are more
   severe than urban ones (29.4% vs 21.9%). Single carriageways (26.2%) are more severe than
   roundabouts (17.1%), slip roads (18.1%) and one-way streets (18.0%)
   (`figures/eda_04_road_factors.png`).

6. **Junctions.** Collisions away from a junction are slightly more severe (26.2%) than those at
   T-junctions (23.6%), crossroads (23.3%) or complex junctions (22.6%). Junction type is
   therefore a weaker separator than speed limit or road type
   (`figures/eda_04_road_factors.png`).

7. **Lighting matters, weather and surface barely do.** Darkness without street lighting is
   associated with a severe rate of 34.5% [33.7, 35.4], compared with 23.3% in daylight and
   25.4% in darkness with lit lamps. Rain (24.4%) and wet roads (25.4%) are close to the overall
   rate of 24.4%, which is consistent with drivers slowing down in poor conditions. High winds
   (27–30%) and fog (27.1%) show modest increases, but these groups are small
   (`figures/eda_05_environment_factors.png`).

8. **Missing values are informative.** Collisions with missing or unknown road type, junction
   detail, weather or road surface have very low severe rates (7.8%, 13.0%, 11.1% and 7.3%
   respectively). Incomplete records are probably dominated by minor, self-reported collisions.
   Missingness should therefore be encoded as an explicit level rather than imputed away
   (`figures/eda_04_road_factors.png`, `figures/eda_05_environment_factors.png`).

9. **Road-user mix.** Collisions with a pedestrian casualty (32.2% vs 22.7%) or a motorcycle
   (32.9% vs 22.7%) are much more often severe. Single-vehicle collisions are also more severe
   than two-vehicle collisions (32.1% vs 20.4%). Pedal-cycle involvement (25.6% vs 24.2%) and
   heavy goods vehicles (27.2% vs 24.3%) raise the rate only modestly
   (`figures/eda_06_vehicle_mix.png`).

10. **Geography is confounded by force reporting practice.** The severe rate is lowest around
    London and in south-west England and highest in Scotland and northern England. Collisions
    north of latitude 55.8° have a severe rate of 43.5% [42.3, 44.6], against 23.7% further
    south (`figures/eda_07_geo_hexbin.png`). Police-force rankings largely follow the recording
    system. Forces using injury-based recording in both years pool to 28.1%, forces using
    severity-based recording in both years pool to 18.9%, and the Metropolitan Police
    (severity-based, n = 43,574) ranks last at about 16%
    (`figures/eda_08_police_force_rank.png`). Geographic and force effects therefore combine
    road risk with measurement practice, and they should be interpreted with care in SHAP
    analyses (sensitivity S3).

11. **Police attendance is leakage.** An officer attended 86% of severe collisions but only 64%
    of slight ones. The severe rate is 30.2% when an officer attended, 14.8% when no officer
    attended and 9.7% for self-completion reports. Attendance is decided after the outcome is
    known, so it is excluded from the features and used only as an evaluation stratifier
    (`figures/eda_10_police_attendance.png`).

12. **Schema drift.** Several fields are being phased out. In the label-free view, the
    `*_historic` fields and `special_conditions_at_site` go from 2–10% missing in 2023 to 61–68%
    in 2024 and 87–89% in 2025. The historic journey-purpose field goes from 58% to 93%. `age_of_vehicle` is 100% missing in 2025, and the
    driver/casualty distance bandings are 100% missing in 2023. In contrast, the kept features
    stay stable across all three years (2–8% missing at collision level). This supports the
    feature exclusions in the leakage audit (`figures/eda_11_missingness_drift.png`).
