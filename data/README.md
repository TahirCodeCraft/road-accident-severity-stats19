# Data

`raw/` holds the files used in this study, downloaded from the Department for Transport
"Road safety open data" page on 12 November 2026:
<https://www.gov.uk/government/statistical-data-sets/road-safety-open-data>

| File | Content |
|---|---|
| `dft-road-casualty-statistics-collision-{2023,2024,2025}.csv` | one row per collision |
| `dft-road-casualty-statistics-vehicle-{2023,2024,2025}.csv` | one row per vehicle |
| `dft-road-casualty-statistics-casualty-{2023,2024,2025}.csv` | one row per casualty |
| `data-guide-2025.xlsx` | DfT code lists for every categorical field |

DfT may revise published years; the copies here are the exact inputs behind the reported results.

`processed/` is not committed. `scripts/01_build_dataset.py` and `scripts/03_build_features.py`
rebuild it from `raw/`.

**Licence and attribution.** Contains public sector information licensed under the Open
Government Licence v3.0 (<https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>).
Source: Department for Transport, STATS19 road safety data.
