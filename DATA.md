# Data sources

Everything in `data/curated/` is derived from public sources. Where a licence requires it, attribution is below.

| Files | What | Source | Licence |
|---|---|---|---|
| `forecast-v1/anchor-*.csv`, `flows-final-pair-empirical.csv`, `ballot-top-*.csv`, `seat-context-2026.csv`, `backtest-inputs/seat-context-*.csv`, `vic2026_la/` | Seat results 2002-2022 (and the 2023 Narracan supplementary election), distributions of preferences, ballot order, candidates | Victorian Electoral Commission results and candidate pages, vec.vic.gov.au | CC BY 4.0, © Victorian Electoral Commission |
| `forecast-v1/on-federal-2025-by-district.csv`, `federal-tpp-relative-swing.csv`, parts of `on-geography-2026.csv` and `backtest-inputs/seat-vars-*.csv` | Federal polling-place results 2019-2025 placed on state districts | Australian Electoral Commission, aec.gov.au | CC BY 4.0, © Commonwealth of Australia (AEC) |
| `forecast-v1/seat-shifts-2026.csv`, `seat-index-*.csv`, `backtest-inputs/seat-shifts-*.csv` | 2021 Census profiles by district | Australian Bureau of Statistics, abs.gov.au | CC BY 4.0, © Commonwealth of Australia (ABS) |
| `polls/polls-2026-20261122.csv`, `forecast-v1/polls-20*-all.csv`, `region-split-polls-2026.csv` | Published opinion polls | Wikipedia poll lists for each Victorian election, and the pollsters' published reports | Wikipedia text CC BY-SA 4.0 |
| `forecast-v1/mrp-yougov-2026-by-district.csv` | YouGov MRP seat estimates, June-July 2026 | YouGov / Common Threads published report | Published figures, cited |
| `forecast-v1/on-poll-error-history.json`, `poll-error-calibration.json`, `swing-dispersion*.json` | Calibrations computed from the results and polls above | this project | MIT, with the sources above |
| `embedding_features/` | Eight geodemographic embedding components per district | the author's own geospatial embedding model | MIT |
| `forecast-v1/credible-ind-2026.csv`, `backtest-inputs/credible-ind-2022.csv` | Climate 200-backed independents | news reports (Crikey, ABC) and Climate 200 | facts, cited |

Seat results and candidates: © Victorian Electoral Commission, CC BY 4.0. This model is not endorsed by the VEC.
