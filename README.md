# Victoria 2026 seat model (Model v1.25)

A seat-by-seat forecast of the 28 November 2026 Victorian Legislative Assembly election: every published poll,
the 2022 seat results, the candidates on the ballot and an instant-runoff count in all 88 districts, simulated
10,000 times. This repository holds the model core, the inputs it reads, the settings of the released edition and
the backtests on the five previous Victorian elections, so that anyone can rerun it or test their own model against it.

Published forecast: <https://spatialsearch.io/elections/> (forecast as of 22 November 2026, Model v1.25).

## Headline (22 November 2026, 37 polls to 20 November)

| | Mean seats | 90% range | Majority (45 of 88) |
|---|---|---|---|
| Coalition (Liberal + National) | 51.3 | 35-67 | 74.8% |
| Labor | 22.2 | 8-39 | 1.1% |
| One Nation | 8.6 | 1-21 | |
| Greens | 5.3 | 3-9 | |
| Independents | 0.6 | 0-2 | |

Hung parliament 24%. Seat-by-seat win chances, primary shares and final-pair chances: `reference/v1.25-2026-11-22/seats.csv`.

## Run it

Python 3.11+ with numpy, pandas and pyarrow (`pip install -r requirements.txt`).

```bash
# the 2026 forecast as released (about 3 minutes; reproduces reference/v1.25-2026-11-22/seats.csv exactly)
python scripts/run_forecast_v1.py --as-of 2026-11-22 --draws 10000 --config configs/model-v1.25.json \
    --polls data/curated/polls/polls-2026-20261122.csv --out runs/v1.25

# backtests: 2006, 2010, 2014, 2018 and 2022, each forecast from the election before it (about 1 minute a seed)
python scripts/run_backtests.py --seeds 1      # --seeds 6 for the published averages
```

Outputs go to `runs/` (`seats.csv`: one row per district with `primary_<party>_mean`, `win_<party>`, final-pair
probabilities; `edition.json`: statewide seat distributions, majority and hung probabilities, two-party vote).

## How it works

1. **Polls.** Every published statewide poll since the 2022 election, weighted by sample size with a design effect,
   pollster house effects shrunk toward zero, a 35-day kernel-weighted local-linear trend, movement to polling
   day and one shared election-wide poll error per simulation (calibrated on past Victorian, federal and other-state
   misses; One Nation's from its own record).
2. **Seats.** Each district starts from its 2022 result (Narracan from its 2023 supplementary election). Labor and
   Coalition swings are uniform in points; Greens and others half uniform, half proportional; regional and seat noise
   (larger in country seats), sitting-member and retirement effects, a federal-swing term, and a Greens lean toward
   seats with more residents aged 20-34. One Nation's statewide vote is spread over seats by a stacked pattern of its
   2025 and 2022 federal vote, the 2022 right-minor state vote, geodemographic embedding components and the YouGov
   MRP, with seat polls as extra observations.
3. **Candidates.** The final VEC ballot decides who stands. Returning independents start from their own previous vote;
   new independents from a lognormal share (higher in safe and open seats); Climate 200-backed new independents start
   higher, with the extra votes taken from Labor, the Greens and minor parties (v1.25).
4. **Preferences.** A full instant-runoff count in every seat in every simulation, with measured Victorian flows,
   origin-tracked parcels, how-to-vote card orders as registered, and explicit final-pair rates (Coalition v One Nation
   at South Australia 2026 rates since v1.24).
5. **Results.** Winner probabilities, seat counts, majority and hung probabilities all come from the same 10,000
   simulated elections, so seats move together the way they did historically.

Every scale is a declared assumption in `DEFAULTS` (top of `src/vicmodel/forecast_v1.py`) or in the config.

## Backtests (Model v1.25 settings, six seeds)

| Election | Winner Brier | Log loss | Winners right |
|---|---|---|---|
| 2006 | 0.105 | 0.179 | 82 / 88 |
| 2010 | 0.090 | 0.154 | 83 / 88 |
| 2014 | 0.112 | 0.190 | 67 / 88 |
| 2018 | 0.184 | 0.317 | 76.5 / 88 |
| 2022 | 0.105 | 0.179 | 74 / 88 |
| mean | 0.119 | 0.204 | 76.5 |

Brier is summed over party groups (Labor, Coalition, Greens, One Nation, independent, other) and averaged over seats;
log loss floors the winner's probability at 0.005. 2022 crosses a redistribution, so only name-matched districts are
scored. The backtests are honest about timing only in part: polls, results and candidates are as known before each
election, but some coefficients were chosen on these same five elections (see `CHANGES.md`).

## Release history

`CHANGES.md` lists what each release since v1.19 changed and what it did to the backtests.

## Data and licences

Code: MIT (`LICENSE`). Inputs are derived from public sources; see `DATA.md` for sources, attributions and licences
(VEC, AEC and ABS material is CC BY 4.0; the poll list is from Wikipedia, CC BY-SA 4.0).
