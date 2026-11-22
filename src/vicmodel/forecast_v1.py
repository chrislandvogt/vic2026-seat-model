"""Model v1: a complete, simple, declared-assumption forecast of the 2026 Victorian Legislative Assembly.

Pipeline (one call to ``run_forecast``):

1. Poll aggregate: every published 2026 statewide poll, multinomial precision with a design effect,
   pollster house effects shrunk toward zero, kernel-smoothed trend, movement to election day and a shared
   election-wide bias draw. Produces draws of statewide election-day shares for ALP, LNP, GRN, ON, OTH.
2. Seat primaries: 2022 seat results (Narracan from its 2023 supplementary election) plus proportional
   swing in the non-One-Nation remainder, regional and seat innovations, and a One Nation seat share
   allocated from the statewide ON draw by the 2025 federal ON district pattern (shrunk toward uniform)
   with the Roy Morgan three-seat poll as an extra observation.
3. Candidates: the ABC provisional roster decides which of LIB/NAT stand and whether a 2022 independent
   with a material vote recontests. Majors are assumed to stand everywhere before nominations close.
4. Preferences: aggregate instant-runoff count per seat per draw with origin-tracked parcels, transfer
   weights with Beta uncertainty drawn once per simulated election, and explicit final-pair preference rates
   for the contests that decide seats, including One Nation finalists.
5. Outputs: seat primaries, winner probabilities, final-pair probabilities, Labor-Coalition 2PP where both
   stand, seat-count distributions and majority probabilities. Every number comes from the same draws.

All scales are declared assumptions listed in ``DEFAULTS``; the methods note documents them.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PARTIES = ["ALP", "LNP", "GRN", "ON", "OTH"]            # statewide poll parties
GROUPS = ["ALP", "LIB", "NAT", "GRN", "ON", "IND", "OTH"]  # seat-level groups

DEFAULTS = {
    "election_day": "2026-11-28",
    "design_effect": 1.5,          # nominal n / design effect = effective n
    "default_sample": 1000,
    "rounding_sd": 0.003,          # published whole-percent rounding
    "house_prior_sd": 0.015,       # shrinkage of pollster house effects toward zero
    "trend_bandwidth_days": 21.0,  # Gaussian kernel for the poll trend. v1.18: was 35; 21 scores better in both backtests
                                   # for three seeds (Brier -0.002, reports/backtest-sweep-20261115.json)
    "movement_sd_per_30_days": 0.015,  # unobserved movement between last poll and election day
    "bias_sd": 0.02,               # shared election-wide polling error per party
    "regional_sd": 0.02,           # regional deviation from statewide swing, per party per draw
    "seat_sd": 0.03,               # seat innovation, per party per draw
    "on_pattern_shrink": 1.0,      # the stacked ON pattern is already a shrunk regression; no extra pull toward uniform (v1.6; was 0.8)
    "on_regional_log_sd": 0.10,
    "on_seat_log_sd": 0.30,       # leave-one-out residual sd of the embedding model of federal ON geography
    "on_floor": 0.03,              # One Nation contests every seat; floor on its seat share
    "ind_retention_log_sd": 0.30,  # recontesting independent: 2022 share times lognormal
    "ind_min_share_2022": 0.08,    # independents below this fold into OTH
    "new_ind_share": 0.03,         # a new independent on the roster with no history
    "flow_concentration": 60.0,    # Beta concentration for transfer-rate uncertainty
    "seat_flow_sd": 0.04,          # seat-level noise on final-pair preference rates
    "draws": 2000,
    "seed": 20261104,
    "majority": 45,
    # Model v1.1 additions
    "model_version": "1.11",
    "on_geography_file": "data/curated/forecast-v1/on-geography-2026.csv",   # stacked ON pattern (federal 2025/2022, 2022 right-minor vote, embedding, MRP)
    "on_pattern_log_sd_floor": 0.18,   # the MRP is a model, not an observation: do not let the stack claim better than this
    "seat_context_file": "data/curated/forecast-v1/seat-context-2026.csv",   # rosters, retirements, sophomores, independents
    "sophomore_effect": 0.020,   # first-term member's party gains (personal vote built since first election). v1.10: measured on the
    "sophomore_sd": 0.010,       # holder's 2CP, 2014->2018 and 2018->2022, 138 classic seats: +2.2 (se 0.8); succession +2.2, party-change +4.2
    "on_flow_lean_slope": 0.46,
    "swing_scale": "mixed",     # half additive, half log-odds: additive scores best on 2014->2018 and 2018->2022 backtests, but 2026's
                                # Labor swing (about -12 pp) is outside that range and pure additive drives small bases to zero (v1.7)  # ON->Coalition (over Labor) rate rises 0.46 per unit of seat Coalition 2022 share above the state mean (federal 2025 divisions, r=0.78)
    "mrp_file": "data/curated/forecast-v1/mrp-yougov-2026-by-district.csv",   # YouGov MRP June-July 2026, 88 seats
    "on_pattern_weights": {"federal": 0.3, "embedding": 0.3, "mrp": 0.4},
    "on_seat_log_sd_with_mrp": 0.24,   # three partly independent pattern sources; 0.30 without the MRP
    "swing_dispersion_file": "data/curated/forecast-v1/swing-dispersion-pooled-2022.json",   # v1.19: seat/regional sd pooled over five pairs 2002->2022 (was 2014->2018 only)
    "poll_error_file": "data/curated/forecast-v1/poll-error-calibration.json",   # Victorian final-fortnight poll error
    "bias_mean_shrink": 0.0,     # share of the Victorian historical mean poll error applied as a correction. v1.18: was 0.5;
                                 # the Greens overstatement (3.8 pp 2010) has faded to ~0 and Labor's sign flipped in 2022, and
                                 # 0 scores best in both backtests for three seeds (Brier -0.007/-0.006)
    "bias_mean_shrink_on": 0.5,  # One Nation's correction from other jurisdictions (on_poll_error_file), unchanged
    "bias_sd_floor": 0.015,
    "bias_sd_new_party": 0.03,   # One Nation has no Victorian poll-error history
    "retirement_effect": 0.015,  # share lost by the party of a retiring member (v1.10 check: -1.5, se 0.9, same 138 seats)
    "retirement_sd": 0.01,
    "embedding_file": "data/curated/embedding_features/northern-metropolitan_district_features.parquet",
    "on_embedding_weight": 0.5,   # weight of the embedding-predicted ON pattern vs the observed federal pattern
    "on_ridge_lambda": 1.0,
    "sim_rho": 0.5,               # share of seat innovation variance that is shared with similar districts
    "sim_bandwidth": 2.5,         # Gaussian kernel width in standardised embedding-component distance
    "flows_file": "data/curated/forecast-v1/flows-final-pair-empirical.csv",
    "flow_min_conc": 8.0,
    "flow_max_conc": 200.0,
    # Model v1.8 additions (VEC final roster, 9 November)
    "departed_ind_to_parties": 0.68,   # share of a non-recontesting 2022 independent's vote (>= ind_min_share_2022) that returns to
                                       # Labor/Coalition/Greens in proportion to their seat shares; the rest stays with minor
                                       # parties and new independents (11 Victorian cases 2014->2018 and 2018->2022: 0.21/0.38/0.08)
    "absent_party_base_file": "data/curated/forecast-v1/anchor-2018.csv",   # base for a major party standing in 2026 with no
                                       # 2022 vote in the seat (Narracan: 2023 supplementary had no Labor candidate)
    # Model v1.9 addition (VEC ballot draw, 9 November): the top-of-ballot (donkey vote) gain, in primary-share points,
    # taken from the other candidates in proportion. Victorian 2006-2022 results, same party in the same seat in
    # consecutive elections: +2.4 (se 0.3) when it moves to the top; minor/independent candidates at the top +1.6 (0.3).
    # Conditional on primaries the order between the final two adds nothing to the 2CP (0.0, se 0.6).
    "ballot_top_gain": 0.024,          # ALP, LIB, NAT, GRN, ON
    "ballot_top_gain_minor": 0.016,    # IND, OTH
    "ballot_base_file": "data/curated/forecast-v1/ballot-top-2022.csv",   # top of the ballot in the anchor election
    "ballot_now_file": "data/curated/forecast-v1/ballot-top-2026.csv",    # top of the ballot in the forecast election
    # Model v1.10 addition (9 November): federal relative swing. A district's state Labor-v-Coalition 2CP swing, relative
    # to the statewide swing, followed the federal relative 2PP swing in the same district over the preceding term:
    # 2018->2022 state on 2019->2022 federal (63 classic name-matched seats), slope 0.64 (se 0.11), R2 0.34; 0.56 with
    # retirement/sophomore controls. Applied as a transfer between the Labor and Coalition compositions of
    # coef x federal relative swing 2022->2025 (AEC booths on current districts), coef drawn N(0.5, 0.15).
    "federal_swing_file": "data/curated/forecast-v1/federal-tpp-relative-swing.csv",
    "federal_swing_col": "rel_swing_2022_2025_classic",   # withheld where the main federal division had a non-classic final two
    "federal_swing_coef": 0.5,
    "federal_swing_coef_sd": 0.15,
    "federal_swing_classic_only": True,   # apply only where the anchor election's final pair was Labor v Liberal/National: in
                                          # Greens and independent contests the Labor-v-Coalition shift moved Labor against the
                                          # wrong opponent (2022 backtest Brier 0.200 -> 0.241 without this restriction)
    # Model v1.17 addition (15 November): shared demographic swing factors (apply_seat_factors). Off by default.
    "seat_index_file": None,          # e.g. data/curated/forecast-v1/seat-index-2026.csv (scripts/check_seat_covariates_20261115.py)
    "seat_factors": None,             # {index column: {"mean": pp/100 per sd, "sd": ...}}
    "seat_factor_classic_only": False,
    # Model v1.12 additions (10 November). All default off so backtests and earlier configs are unchanged.
    # (a) Coalition preference decision (data/curated/forecast-v1/preference-decisions-2026.json): one shared draw per
    #     simulated election; if the Coalition recommends One Nation above Labor, Coalition preferences in Labor-v-One-Nation
    #     counts go lnp_to_alp_vs_on_recommended to Labor (SA 2026 measured 0.33-0.35), otherwise lnp_to_alp_vs_on_open
    #     (0.47, implied by Victorian poll pairings). Nationals resolved on 9 November, so NAT-only seats always recommend.
    "coalition_recommends_on_p": None,
    "lnp_to_alp_vs_on_recommended": 0.35,
    "lnp_to_alp_vs_on_open": 0.47,
    # (b) One Nation's shared poll error from other jurisdictions (QLD 2017/2024, federal 2025, SA 2026)
    "on_poll_error_file": None,
    "bias_sd_floor_on": 0.02,
    # (c) Melbourne/country balance of each party pulled toward published poll crosstabs, precision-weighted against the
    #     model's own uncertainty about that balance
    "region_split_file": None,
    "region_split_parties": ["ALP", "LNP", "GRN", "ON"],
    "region_split_half_life_days": 90.0,
    "region_split_definition_sd": 0.03,   # log-ratio floor: pollsters define Melbourne/country differently
    # Model v1.13 additions (12 November): registered cards, announcements and a seat poll. All default off; they draw
    # from their own stream (seed, 113), so with them off a v1.12 config reproduces its draws exactly.
    # (a) Liberal order of Labor versus the Greens: one shared draw per simulated election. Victorian distributions of
    #     Liberal ballots in Labor-v-Greens counts (inner-city seats): 2014 and 2018, Liberals recommending Labor above
    #     the Greens, 0.69 and 0.57 to Labor (mean 0.63); 2022, recommending the Greens above Labor, 0.35. Open ticket
    #     (no card) 0.49, the midpoint, which implies about 28% of Liberal voters follow the card on this choice.
    "lib_alp_over_grn_p": None,
    "lnp_to_alp_vs_grn_directed": 0.63,
    "lnp_to_alp_vs_grn_reversed": 0.35,
    "lnp_to_alp_vs_grn_open": 0.49,
    "lnp_alp_grn_concentration": 60.0,   # between-election sd about 0.06 within a regime (0.69 vs 0.57)
    "lib_open_alp_grn_seats": [],        # seats with a Liberal open ticket (or no Liberal cards): rate to Labor = open
    # v1.14: when the registered Liberal cards show the order, lib_alp_over_grn_p is 1.0 and this records what was read,
    # e.g. {"as_of": "12 November", "seats_with_cards": 72, "labor_above_greens": 72, "no_card_yet": [...]}
    "lib_cards_registered": None,
    # (b) seat-level Coalition rate to Labor in Labor-v-One-Nation counts where the seat's Liberal card differs from the
    #     statewide decision, e.g. One Nation's candidate placed last ({"preston": 0.59}) or no Liberal cards (0.47).
    "seat_lnp_alp_vs_on_rate": {},
    # (c) fixed seat-level shifts of a final-pair rate: {district: [[origin, first, second, shift_toward_first], ...]},
    #     e.g. a Labor open ticket moves Labor ballots from the directed rate to the unguided rate.
    "seat_flow_adjust": {},
    # (d) candidates disendorsed after the ballot draw stay on the ballot under the party name with no campaign: the
    #     group's seat share falls by a declared relative cut (Lyons 2019 Liberal -6%, Melbourne 2019 Labor -12%).
    "disendorsed": {},                   # district -> group
    "disendorsed_share_cut": 0.10,
    "disendorsed_share_cut_sd": 0.05,
    # (e) further seat polls as One Nation ratio observations, same form as SEAT_ON_OBSERVATIONS
    "extra_seat_on_observations": [],
    # Model v1.15 (13 November). Default off. Parties whose Melbourne/country correction falls on the rural seats only:
    # the provincial cities (PROVINCIAL_CITY_DISTRICTS) take Melbourne's factor and the rural seats carry the whole
    # country correction, so the country total, the Melbourne seats and the statewide totals are as before. The poll
    # crosstabs count the provincial cities as country; past results and the YouGov MRP put the model's Labor shortfall
    # in the rural seats (reports/country-factor-check-20261113.json).
    "region_split_rural_only_parties": [],
    # Model v1.20 candidates (16 November). All default off.
    "ind_retention_log_mean": None,  # mean of log(returning independent's share / previous share); None keeps the mean at 1
    "new_ind_log_sd": None,          # log sd of the leading new independent's share (None: the pre-v1.20 narrow draw)
    "new_ind_extra_log": 0.0,        # added to the log median per further new independent on the ballot
    "new_ind_country_log": 0.0,      # added to the log median outside Melbourne
    "seat_context_force": False,     # read seat_context_file even when the One Nation pattern files are off (backtests)
    "ind_base_top_only": False,      # a recontesting independent's base is that candidate's own anchor share
    # Seat-level shifts from public information, each {"file", "col", "target", "coef", "coef_sd", "center", "classic_only"}.
    # target "ALP_LNP" moves coef x value of the seat's vote from the Coalition to Labor; "GRN", "OTH", "ALP" or "LNP" adds
    # it to that group and takes it from the others in proportion. One coefficient is drawn per simulated election.
    "seat_shifts": [],
    # Tally Room ideas (19 November 2026, docs/tallyroom-ideas-20261119.md). All default off; nothing adopted.
    # (a) Final-pair preference rates replacing the declared and empirical ones, e.g. the South Australian 2026
    #     measurements: {"LNP-ON": {"ALP": 0.69, "GRN": 0.65}, ...}; rate toward the first group named. Beta draws with
    #     concentration final_pair_override_concentration (None: flow_concentration).
    "final_pair_rates_override": None,
    "final_pair_override_concentration": None,
    # (b) First-preference moves between groups in named seats, e.g. a party whose name resembles another's drawing
    #     votes from it: {district: [[from_group, to_group, mean_share, sd_share], ...]}. Own random stream.
    "seat_primary_moves": None,
    # Gender hypotheses (19 November 2026, docs/gender-20261119.md). Default off; nothing adopted.
    # A change in a party's gender gap (women's share minus men's, in points) since the anchor election moves each seat by
    # gap x (the seat's women's share of adults - the formal-weighted mean): {"ON": [mean_pts, sd_pts], "LNP": ...}.
    # Groups: ALP, LNP (Liberal + National), GRN, ON, IND, OTH. The women's shares are read from gender_gap_file/_col.
    "gender_gap_change": None,
    "gender_gap_file": None,
    "gender_gap_col": "f_adult",
    # Leadership-approval hypotheses (19 November 2026, docs/leadership-approval-20261119.md). Default off; nothing adopted.
    # A statewide first-preference transfer to the government party from the main opposition (negative: the government
    # does worse than its polls), drawn once per simulated election as Normal(leader_state_shift, leader_state_shift_sd);
    # with a zero shift and a positive sd it only widens the government-v-opposition uncertainty. Own random stream.
    "leader_state_shift": None,
    "leader_state_shift_sd": 0.0,
    "leader_gov_party": "ALP",     # the party in office going into the election (ALP or LNP); the other one loses/gains
}

# Districts in the three non-metropolitan Legislative Council regions that lie mostly inside Greater Melbourne
# (ABS GCCSA), used to split seats into Melbourne and country for the poll-crosstab calibration.
GREATER_MELBOURNE_IN_COUNTRY_REGIONS = {"evelyn", "hastings", "monbulk", "mornington", "nepean", "pakenham", "melton",
                                        "yan-yean"}


def is_melbourne(seat) -> bool:
    return "METROPOLITAN" in str(seat.region) or seat.district_id in GREATER_MELBOURNE_IN_COUNTRY_REGIONS


# v1.15: country districts centred on Geelong, Ballarat and Bendigo, plus Macedon. Ripon, which takes in part of
# Ballarat's fringe, stays rural.
PROVINCIAL_CITY_DISTRICTS = {"geelong", "lara", "bellarine", "south-barwon", "wendouree", "eureka", "bendigo-east",
                             "bendigo-west", "macedon"}

# Transfer weights when an origin group's ballots leave an excluded candidate and more than two candidates
# remain. Values are relative weights over the remaining candidates. Sources: 2022 Victorian distribution
# ledgers (data/curated/forecast-v1/flows-2022-empirical.csv), 2025 federal Victorian ON flows (ON->LNP 75%),
# and declared judgement for pairs with no Victorian evidence.
TRANSFER_WEIGHTS = {
    "ALP": {"LIB": 0.30, "NAT": 0.30, "GRN": 0.55, "ON": 0.10, "IND": 0.60, "OTH": 0.25},
    "LIB": {"ALP": 0.40, "NAT": 0.90, "GRN": 0.25, "ON": 0.45, "IND": 0.55, "OTH": 0.30},
    "NAT": {"ALP": 0.30, "LIB": 0.90, "GRN": 0.15, "ON": 0.45, "IND": 0.55, "OTH": 0.30},
    "GRN": {"ALP": 0.80, "LIB": 0.18, "NAT": 0.18, "ON": 0.06, "IND": 0.50, "OTH": 0.30},
    "ON": {"ALP": 0.22, "LIB": 0.60, "NAT": 0.60, "GRN": 0.08, "IND": 0.45, "OTH": 0.40},
    "IND": {"ALP": 0.45, "LIB": 0.40, "NAT": 0.40, "GRN": 0.25, "ON": 0.25, "OTH": 0.35},
    "OTH": {"ALP": 0.28, "LIB": 0.30, "NAT": 0.30, "GRN": 0.20, "ON": 0.35, "IND": 0.40},
}

# Final-pair preference rates: probability that an origin group's ballot goes to the FIRST member of the
# pair. Pairs are stored with Coalition as LNP; LIB and NAT map to LNP. Each (origin, pair) gets a Beta
# draw per simulated election with concentration ``flow_concentration``.
FINAL_PAIR_RATES = {
    ("ALP", "LNP"): {"GRN": 0.80, "ON": 0.26, "IND": 0.55, "OTH": 0.47},
    ("ALP", "GRN"): {"LNP": 0.50, "ON": 0.70, "IND": 0.50, "OTH": 0.50},
    ("LNP", "GRN"): {"ALP": 0.20, "ON": 0.85, "IND": 0.45, "OTH": 0.45},
    # One Nation finalist rates from SA 2026 (data/curated/forecast-v1/sa-2026-one-nation-flows.json), blended
    # toward Victorian evidence where it exists (Roy Morgan pairings, Nepean 2026).
    ("ALP", "ON"):  {"LNP": 0.40, "GRN": 0.85, "IND": 0.56, "OTH": 0.56},
    ("LNP", "ON"):  {"ALP": 0.80, "GRN": 0.92, "IND": 0.55, "OTH": 0.45},
    ("ALP", "IND"): {"LNP": 0.20, "GRN": 0.55, "ON": 0.25, "OTH": 0.40},
    ("LNP", "IND"): {"ALP": 0.20, "GRN": 0.15, "ON": 0.775, "OTH": 0.40},  # ON->LIB 0.775 at Nepean 2026
    ("GRN", "IND"): {"ALP": 0.55, "LNP": 0.35, "ON": 0.30, "OTH": 0.45},
    ("GRN", "ON"):  {"ALP": 0.85, "LNP": 0.40, "IND": 0.55, "OTH": 0.45},
    ("IND", "ON"):  {"ALP": 0.75, "LNP": 0.55, "GRN": 0.85, "OTH": 0.50},
}

# Roy Morgan 21-26 October SMS poll, combined Hastings/Morwell/Pakenham primary table (283 nominal across four
# seats including Bass). Treated as one soft observation of the ON seat ratio for those three seats.
MORGAN_SEAT_POLL = {"districts": ["hastings", "morwell", "pakenham"], "on_share": 0.39, "state_on_at_time": 0.22,
                    "weight": 0.5}
# Nepean by-election, 2 May 2026: One Nation 24.5% with statewide polling about 23% at the time (Labor absent).
SEAT_ON_OBSERVATIONS = [MORGAN_SEAT_POLL,
                        {"districts": ["nepean"], "on_share": 0.245, "state_on_at_time": 0.23, "weight": 0.5}]


def _as_date(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


@dataclass
class PollAggregate:
    parties: list[str]
    level: np.ndarray            # election-day expected share per party
    level_sd: np.ndarray         # uncertainty of the smoothed level at the last fieldwork date
    house: dict                  # pollster -> array of house effects
    last_fieldwork: date
    days_ahead: int
    n_polls: int
    detail: pd.DataFrame = field(repr=False, default=None)


def aggregate_polls(polls: pd.DataFrame, as_of: date, cfg: dict) -> PollAggregate:
    """Kernel-smoothed poll trend with shrunk house effects. Returns the level on the latest fieldwork date."""
    df = polls.copy()
    df = df[df["fieldwork_end"].notna() & (df["fieldwork_end"].astype(str) != "")]
    df["end"] = df["fieldwork_end"].map(_as_date)
    df["start"] = df["fieldwork_start"].map(_as_date)
    df = df[df["end"] <= as_of].copy()
    df["mid"] = [s + (e - s) / 2 for s, e in zip(df["start"], df["end"])]
    df["n"] = pd.to_numeric(df["sample"], errors="coerce").fillna(cfg["default_sample"])
    cols = {"ALP": "alp", "LNP": "lnp", "GRN": "grn", "ON": "onp"}
    y = np.zeros((len(df), len(PARTIES)))
    for j, p in enumerate(PARTIES[:-1]):
        y[:, j] = pd.to_numeric(df[cols[p]], errors="coerce").to_numpy() / 100.0
    ind = pd.to_numeric(df["ind"], errors="coerce").fillna(0).to_numpy() / 100.0
    oth = pd.to_numeric(df["oth"], errors="coerce").fillna(0).to_numpy() / 100.0
    y[:, 4] = ind + oth
    on_missing = np.isnan(y[:, 3])
    if on_missing.all():
        y[:, 3] = 0.0          # One Nation not reported in this election's polls: treated as absent
    ok = ~np.isnan(y[:, :4]).any(axis=1)
    df, y = df[ok].reset_index(drop=True), y[ok]
    y = y / y.sum(axis=1, keepdims=True)  # published rows sum to 99-101
    n_eff = df["n"].to_numpy() / cfg["design_effect"]
    sd = np.sqrt(y * (1 - y) / n_eff[:, None]) + cfg["rounding_sd"]
    last = max(df["end"])
    t = np.array([(d - last).days for d in df["mid"]], dtype=float)  # <= 0
    bw = cfg["trend_bandwidth_days"]
    houses = df["pollster"].to_numpy()
    uniq = sorted(set(houses))
    house = {h: np.zeros(len(PARTIES)) for h in uniq}
    # 17 Nov experiments (off by default): down-weight a pollster's polls by 1/sqrt(its polls within 28 days), and polls
    # far from the other polls of the same four weeks (The Swing Is On's two dampers). Weights act through the poll sd.
    damp = np.ones(len(df))
    if cfg.get("poll_freq_damping"):
        for i in range(len(df)):
            near = (np.abs(t - t[i]) <= 14) & (houses == houses[i])
            damp[i] /= math.sqrt(near.sum())
    if cfg.get("poll_outlier_damping"):
        for i in range(len(df)):
            near = (np.abs(t - t[i]) <= 14) & (np.arange(len(df)) != i)
            if near.sum() >= 2:
                dev = 100 * np.abs(y[i, :4] - y[near, :4].mean(axis=0)).max()
                damp[i] /= 1 + max(0.0, dev - 3.0) / 5.0
    sd = sd / np.sqrt(damp)[:, None]

    def smooth(resid_adjusted, at_t):
        w = np.exp(-0.5 * ((t - at_t) / bw) ** 2)[:, None] / sd ** 2
        return (w * resid_adjusted).sum(axis=0) / w.sum(axis=0), 1.0 / np.sqrt(w.sum(axis=0))

    for _ in range(12):
        adj = y - np.array([house[h] for h in houses])
        trend_at_each = np.array([smooth(adj, ti)[0] for ti in t])
        resid = y - trend_at_each
        for h in uniq:
            m = houses == h
            prec = (1.0 / sd[m] ** 2).sum(axis=0)
            est = (resid[m] / sd[m] ** 2).sum(axis=0) / prec
            shrink = prec / (prec + 1.0 / cfg["house_prior_sd"] ** 2)
            house[h] = shrink * est
        # centre house effects (poll-weighted) so the trend is not pulled by the pollster mix
        wts = np.array([(houses == h).sum() for h in uniq], dtype=float)
        centre = sum(w * house[h] for w, h in zip(wts, uniq)) / wts.sum()
        for h in uniq:
            house[h] = house[h] - centre
    adj = y - np.array([house[h] for h in houses])
    level, level_sd = smooth(adj, 0.0)
    if cfg.get("trend_local_linear"):
        # 17 Nov experiment (off by default): a kernel-weighted straight line instead of a kernel mean at the latest
        # date (the mean lags a moving trend at the edge of the data), optionally carried `trend_momentum` of the way
        # from the latest fieldwork date to polling day. Variance of a + b*h from weighted least squares.
        h = cfg.get("trend_momentum", 0.0) * max((_as_date(cfg["election_day"]) - last).days, 0) + float(cfg.get("trend_read_offset_days", 0.0))   # offset < 0: read the line that many days before the latest poll (early voting)
        k = np.exp(-0.5 * (t / bw) ** 2)
        X = np.c_[np.ones_like(t), t]
        for j in range(len(PARTIES)):
            w = k / sd[:, j] ** 2
            A = X.T @ (w[:, None] * X)
            if np.linalg.cond(A) > 1e12:
                continue
            Ainv = np.linalg.inv(A)
            beta = Ainv @ (X.T @ (w * adj[:, j]))
            cov = Ainv @ (X.T @ ((k ** 2 / sd[:, j] ** 2)[:, None] * X)) @ Ainv
            v = np.array([1.0, h])
            level[j], level_sd[j] = v @ beta, float(np.sqrt(max(v @ cov @ v, 0.0)))
    level = np.clip(level, 0.0, None)
    level = np.where((level == 0) & (np.arange(len(PARTIES)) != 3), 0.005, level)
    level = level / level.sum()
    election = _as_date(cfg["election_day"])
    detail = df[["pollster", "client", "fieldwork_start", "fieldwork_end", "sample"]].copy()
    for j, p in enumerate(PARTIES):
        detail[p] = y[:, j]
    return PollAggregate(PARTIES, level, level_sd, house, last, (election - last).days, len(df), detail)


def draw_statewide(agg: PollAggregate, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Draws of election-day statewide shares: level + level uncertainty + movement + shared bias."""
    D = cfg["draws"]
    move_sd = cfg["movement_sd_per_30_days"] * math.sqrt(max(agg.days_ahead, 1) / 30.0)
    bias_mean, bias_sd = poll_bias(agg, cfg)
    level_e = rng.normal(0, 1, (D, len(PARTIES))) * agg.level_sd
    move_e = rng.normal(0, move_sd, (D, len(PARTIES)))
    bias_z = rng.normal(0, 1, (D, len(PARTIES)))
    df_t = cfg.get("bias_t_df")
    if df_t:   # sensitivity: a shared Student-t scale per simulated election (polls all miss big together); own stream
        trng = np.random.default_rng([int(cfg["seed"]), 118])
        w = np.sqrt(df_t / trng.chisquare(df_t, D))
        if cfg.get("bias_t_keep_var", True) and df_t > 2:
            w *= math.sqrt((df_t - 2) / df_t)
        bias_z = bias_z * w[:, None]
    e = level_e + move_e + bias_mean + bias_z * bias_sd
    s = agg.level + e
    s = np.clip(s, 0.005, None)
    s[:, agg.level == 0] = 0.0   # parties absent from the polls stay absent
    return s / s.sum(axis=1, keepdims=True)


def leader_state_adjust(state: np.ndarray, cfg: dict) -> np.ndarray:
    """Leadership-approval hypotheses (19 November 2026, off by default): move first preferences between the government
    party (cfg["leader_gov_party"], ALP or LNP) and the other major party in every simulated election, by a draw from
    Normal(leader_state_shift, leader_state_shift_sd) in share units (positive: the government gains). The amount is clipped
    so neither party falls below 0.5%. Own random stream, so no other draw moves; with the options off the input is
    returned unchanged."""
    mean = cfg.get("leader_state_shift")
    sd = float(cfg.get("leader_state_shift_sd") or 0.0)
    if mean is None and sd <= 0:
        return state
    gov = PARTIES.index(cfg.get("leader_gov_party") or "ALP")
    opp = PARTIES.index("LNP" if PARTIES[gov] == "ALP" else "ALP")
    x = np.random.default_rng([int(cfg["seed"]), 141]).normal(float(mean or 0.0), sd, state.shape[0]) if sd > 0 \
        else np.full(state.shape[0], float(mean or 0.0))
    x = np.clip(x, -(state[:, gov] - 0.005), state[:, opp] - 0.005)
    out = state.copy()
    out[:, gov] += x
    out[:, opp] -= x
    return out


@dataclass
class Seat:
    district_id: str
    name: str
    region: str
    formal: int
    share22: dict               # group -> 2022 share
    stands: dict                # group -> bool for 2026
    ind_name: str
    ind_mode: str               # "recontest", "new", "none"
    on_ratio: float             # federal 2025 ON share / statewide federal ON share
    winner22: str = ""
    pc: np.ndarray | None = None   # standardised embedding components
    on_ratio_emb: float = 1.0      # embedding-predicted ON ratio
    retiring_party: str = ""       # ALP/LNP/GRN/IND when the sitting member is flagged as retiring
    on_ratio_mrp: float | None = None   # YouGov MRP seat ON share / its formal-weighted statewide ON share
    on_pattern: float | None = None     # stacked pattern ratio from on-geography file (v1.5)
    on_pattern_log_sd: float | None = None
    sophomore_party: str = ""
    ind_departed: float = 0.0      # 2022 top independent share when that independent does not recontest (v1.8)
    base22: dict | None = None     # 2022 shares with an absent standing major party filled from 2018 (v1.8)
    top_base: str = ""             # group of the anchor election's top-of-ballot candidate (v1.9)
    top_now: dict | None = None    # group -> probability of holding the top of the 2026 ballot (v1.9)
    fed_rel: float = 0.0           # federal ALP 2PP swing in the district minus the statewide swing, last federal term (v1.10)
    classic: bool = True           # anchor election's final pair was Labor v Liberal/National (v1.17)
    ind_top22: float = 0.0         # the anchor election's leading independent's own share (v1.20)
    n_new_ind: int = 0             # independents on the ballot other than a returning one (v1.20)


def load_seats(root: Path, roster_path: Path | None, cfg: dict, anchor_path: Path | None = None,
               on_path: Path | None = None) -> list[Seat]:
    anchor = pd.read_csv(anchor_path or root / "data/curated/forecast-v1/anchor-2022.csv")
    anchor["region"] = anchor["region"].fillna("UNASSIGNED")
    if on_path is None and cfg.get("on_pattern_file", True):
        on_path = root / "data/curated/forecast-v1/on-federal-2025-by-district.csv"
    if on_path and Path(on_path).exists():
        onf = pd.read_csv(on_path).set_index("district_id")
    else:
        onf = pd.DataFrame({"fed2025_on": 1.0}, index=anchor.district_id)
    roster = defaultdict(set)
    ind_names = defaultdict(list)
    if roster_path and roster_path.exists():
        for r in json.loads(roster_path.read_text(encoding="utf-8")):
            roster[r["district_id"]].add(r["party_id"])
            if r["party_id"] == "IND":
                ind_names[r["district_id"]].append(r["candidate_name"])
    fed_mean = float((onf["fed2025_on"].reindex(anchor.district_id).fillna(onf["fed2025_on"].mean()).to_numpy() * anchor.formal).sum() / anchor.formal.sum())
    pcs, emb_ratio = load_embedding_model(root, anchor, onf, fed_mean, cfg)
    mrp_ratio = {}
    mrp_path = root / cfg["mrp_file"] if cfg.get("mrp_file") and cfg.get("on_pattern_file", True) else None
    if mrp_path and mrp_path.exists():
        mrp = pd.read_csv(mrp_path).set_index("district_id")["mrp_on"].reindex(anchor.district_id)
        if mrp.notna().sum() >= 80:
            state_mrp = float((mrp.fillna(mrp.mean()).to_numpy() * anchor.formal).sum() / anchor.formal.sum())
            mrp_ratio = {d: float(v / state_mrp) for d, v in mrp.items() if v == v}
    context = {}
    ctx_path = root / cfg["seat_context_file"] if cfg.get("seat_context_file") and (
        cfg.get("on_pattern_file", True) or cfg.get("seat_context_force")) else None   # force: backtests with a historical roster
    if ctx_path and ctx_path.exists():
        context = pd.read_csv(ctx_path, dtype=str).fillna("").set_index("district_id").to_dict("index")
    geog = {}
    g_path = root / cfg["on_geography_file"] if cfg.get("on_geography_file") and cfg.get("on_pattern_file", True) else None
    if g_path and g_path.exists():
        geog = pd.read_csv(g_path).set_index("district_id").to_dict("index")
    a_prev = None
    if cfg.get("absent_party_base_file") and (root / cfg["absent_party_base_file"]).exists():
        a_prev = pd.read_csv(root / cfg["absent_party_base_file"]).set_index("district_id")
        F22, Fp = anchor.formal.sum(), a_prev.formal.sum()
        st22 = {g: float((anchor[f"share_{g}"] * anchor.formal).sum() / F22) for g in GROUPS}
        stp = {g: float((a_prev[f"share_{g}"] * a_prev.formal).sum() / Fp) for g in GROUPS}
    tops_base, tops_now = {}, defaultdict(list)
    if cfg.get("ballot_base_file") and (root / cfg["ballot_base_file"]).exists():
        tops_base = pd.read_csv(root / cfg["ballot_base_file"]).set_index("district_id")["party_id"].to_dict()
    if cfg.get("ballot_now_file") and (root / cfg["ballot_now_file"]).exists():
        for b in pd.read_csv(root / cfg["ballot_now_file"]).itertuples():
            tops_now[b.district_id].append((str(b.party_id), str(b.candidate_name), float(b.p)))
    fed_rel = {}
    if cfg.get("federal_swing_file") and (root / cfg["federal_swing_file"]).exists():
        fr = pd.read_csv(root / cfg["federal_swing_file"]).set_index("district_id")
        if cfg.get("federal_swing_col") in fr.columns:
            fed_rel = fr[cfg["federal_swing_col"]].dropna().astype(float).to_dict()
    held_map = {"Labor": "ALP", "Liberal": "LNP", "National": "LNP", "Greens": "GRN", "Independent": "IND"}
    retiring = {}
    if "mp" in onf.columns:
        for d, r in onf.iterrows():
            if isinstance(r.get("mp"), str) and "retiring" in r["mp"].lower():
                retiring[d] = held_map.get(str(r.get("held_by", "")).strip(), "")
    cred_names = {}   # credible-independent option (22 Nov 2026): the flagged candidate is the seat's modelled independent
    if cfg.get("cred_ind_file") and (root / cfg["cred_ind_file"]).exists():
        ct = pd.read_csv(root / cfg["cred_ind_file"])
        col = cfg.get("cred_ind_col", "boost")
        if "candidate" in ct.columns and col in ct.columns:
            cred_names = {r.district_id: str(r.candidate) for r in ct.itertuples() if np.isfinite(getattr(r, col))}
    seats = []
    for r in anchor.itertuples():
        share = {g: float(getattr(r, f"share_{g}")) for g in GROUPS}
        parties = roster.get(r.district_id, set())
        if parties:
            lib = "LIB" in parties
            nat = "NAT" in parties
            if not lib and not nat:       # roster incomplete: fall back to 2022 pattern
                lib, nat = bool(r.lib_stood), bool(r.nat_stood)
        else:
            lib, nat = bool(r.lib_stood), bool(r.nat_stood)
        stands = {"ALP": True, "LIB": lib, "NAT": nat, "GRN": True, "ON": bool(cfg.get("on_contests_all", True)), "OTH": True}
        ctx = context.get(r.district_id)
        if ctx:
            stands.update({"ALP": ctx["stands_ALP"] == "1", "LIB": ctx["stands_LIB"] == "1", "NAT": ctx["stands_NAT"] == "1",
                           "GRN": ctx["stands_GRN"] == "1", "ON": ctx["stands_ON"] == "1"})
            ind_names[r.district_id] = [x.strip() for x in ctx["independents"].split(";") if x.strip()]
        # independents: recontest if the 2022 top independent's surname appears on the roster
        ind_mode, ind_name = "none", ""
        if r.top_ind_share >= cfg["ind_min_share_2022"] and isinstance(r.top_ind_name, str) and r.top_ind_name:
            surname = r.top_ind_name.split(",")[0].strip().lower()
            if any(surname in n.lower() for n in ind_names.get(r.district_id, [])):
                ind_mode, ind_name = "recontest", r.top_ind_name
        if ind_mode == "none" and ind_names.get(r.district_id):
            ind_mode, ind_name = "new", ind_names[r.district_id][0]
            want = cred_names.get(r.district_id, "")
            if want:   # the flagged candidate, if on the ballot, is the modelled independent (name only; shares unchanged)
                sur = want.split(",")[0].strip().lower()
                ind_name = next((n for n in ind_names[r.district_id] if sur and sur in n.lower()), ind_name)
        if ind_mode == "none" and cfg.get("assume_ind_recontest", False) and r.top_ind_share >= cfg["ind_min_share_2022"]:
            ind_mode, ind_name = "recontest", str(r.top_ind_name)
        if ctx and ctx.get("ind_recontest"):
            ind_mode, ind_name = "recontest", ctx["ind_recontest"]
        stands["IND"] = ind_mode != "none"
        ind_departed = 0.0
        if cfg.get("departed_ind_to_parties", 0.0) > 0 and ind_mode != "recontest" and r.top_ind_share >= cfg["ind_min_share_2022"]:
            ind_departed = float(min(r.top_ind_share, share["IND"]))
        base22 = None
        if a_prev is not None and r.district_id in a_prev.index:
            for g, members in (("ALP", ("ALP",)), ("GRN", ("GRN",))):
                if stands.get(g) and share[g] < 0.005 and a_prev.loc[r.district_id][f"share_{g}"] > 0.02:
                    x = float(a_prev.loc[r.district_id][f"share_{g}"]) * st22[g] / max(stp[g], 1e-6)
                    b = dict(base22 or share)
                    rest = sum(v for k, v in b.items() if k != g)
                    base22 = {k: (x if k == g else v * (1 - x) / rest) for k, v in b.items()}
        top_now = None
        if tops_now.get(r.district_id):
            top_now = defaultdict(float)
            for pid, cname, p in tops_now[r.district_id]:
                g = pid if pid in GROUPS else "OTH"
                if g == "IND":   # only the modelled independent is the IND group; other independents sit in OTH
                    sur = cname.split(",")[0].strip().lower()
                    g = "IND" if ind_mode != "none" and sur and sur in ind_name.lower() else "OTH"
                if g in ("LIB", "NAT") and not stands.get(g):
                    g = "OTH"
                top_now[g] += p
            top_now = dict(top_now)
        tb = tops_base.get(r.district_id, "")
        top_base = (tb if tb in GROUPS else "OTH") if tb else ""
        seats.append(Seat(r.district_id, r.district, r.region, int(r.formal), share, stands, ind_name, ind_mode,
                          float(onf["fed2025_on"].get(r.district_id, fed_mean)) / fed_mean, str(r.winner_2022),
                          pcs.get(r.district_id), emb_ratio.get(r.district_id, 1.0),
                          (ctx["retiring_party"] if ctx else retiring.get(r.district_id, "")),
                          mrp_ratio.get(r.district_id),
                          (float(geog[r.district_id]["pattern_ratio"]) if r.district_id in geog else None),
                          (float(geog[r.district_id]["pattern_log_sd"]) if r.district_id in geog else None),
                          (ctx["sophomore_party"] if ctx else ""), ind_departed, base22, top_base, top_now,
                          fed_rel.get(r.district_id, 0.0) if (not cfg.get("federal_swing_classic_only", True)
                                                              or str(r.final_pair_2022) in ("ALP-LIB", "ALP-NAT")) else 0.0,
                          str(r.final_pair_2022) in ("ALP-LIB", "ALP-NAT")))
        seats[-1].ind_top22 = float(min(r.top_ind_share, share["IND"]))
        if ctx and ctx.get("region") and r.region == "UNASSIGNED":   # old districts with no current namesake (backtests)
            seats[-1].region = ctx["region"]
        seats[-1].n_new_ind = max(len(ind_names.get(r.district_id, [])) - (1 if ind_mode == "recontest" else 0), 0)
    return seats


def load_embedding_model(root: Path, anchor: pd.DataFrame, onf: pd.DataFrame, fed_mean: float, cfg: dict):
    """Standardised embedding components per district and a ridge prediction of the federal ON ratio from
    those components plus 2022 party shares (leave-one-out R2 0.73 on the 2025 federal pattern)."""
    path = root / cfg["embedding_file"] if cfg.get("embedding_file") else None
    if not path or not path.exists():
        return {}, {}
    emb = pd.read_parquet(path)
    cols = [c for c in emb.columns if c.startswith("pc_")]
    m = anchor[["district_id", "share_ALP", "share_LIB", "share_NAT", "share_GRN"]].merge(emb[["district_id"] + cols], on="district_id", how="left")
    P = m[cols].to_numpy(dtype=float)
    P = np.where(np.isnan(P), 0.0, P)
    P = (P - P.mean(axis=0)) / np.where(P.std(axis=0) > 0, P.std(axis=0), 1.0)
    pcs = {d: P[i] for i, d in enumerate(m.district_id)}
    emb_ratio = {}
    fed = onf["fed2025_on"].reindex(m.district_id)
    if fed.notna().sum() >= 20 and fed.nunique() > 1:
        X = np.c_[P, m[["share_ALP", "share_LIB", "share_NAT", "share_GRN"]].to_numpy(dtype=float)]
        y = np.log(fed.fillna(fed_mean).to_numpy() / fed_mean)
        Xt = np.c_[np.ones(len(y)), X]
        lam = cfg["on_ridge_lambda"]
        beta = np.linalg.solve(Xt.T @ Xt + lam * np.diag([0] + [1] * X.shape[1]), Xt.T @ y)
        pred = np.exp(Xt @ beta)
        emb_ratio = {d: float(pred[i]) for i, d in enumerate(m.district_id)}
    return pcs, emb_ratio


def swing_scales(cfg: dict):
    """Per-party regional and seat innovation sd in PARTIES order, from the measured dispersion file when present."""
    reg = np.full(len(PARTIES), cfg["regional_sd"])
    seat = np.full(len(PARTIES), cfg["seat_sd"])
    path = cfg.get("_root", Path(".")) / cfg["swing_dispersion_file"] if cfg.get("swing_dispersion_file") else None
    if path and path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))["parties"]
        for i, p in enumerate(PARTIES):
            key = "OTH" if p in ("OTH", "ON") else p
            if key in d:
                reg[i] = d[key]["regional_mean_sd"]
                seat[i] = d[key]["within_region_sd"]
            if p in ("OTH", "ON"):   # the measured OTH dispersion is dominated by independents entering and leaving
                reg[i], seat[i] = min(reg[i], 0.02), min(seat[i], 0.05)
    k = float(cfg.get("swing_sd_scale", 1.0))   # sensitivity knob: one election pair measures this spread
    return reg * k, seat * k


def poll_bias(agg: PollAggregate, cfg: dict):
    """Per-party mean correction and sd of the shared election-wide poll error, from Victorian history when present."""
    mean = np.zeros(len(PARTIES))
    sd = np.full(len(PARTIES), cfg["bias_sd"])
    path = cfg.get("_root", Path(".")) / cfg["poll_error_file"] if cfg.get("poll_error_file") else None
    if path and path.exists():
        cal = json.loads(path.read_text(encoding="utf-8"))
        excl = str(cfg.get("poll_error_exclude", ""))
        elections = {e: v for e, v in cal["elections"].items() if e != excl}
        for i, p in enumerate(PARTIES):
            key = p.lower()
            if p in cal["parties"] and elections:
                errs = np.array([elections[e][key] for e in elections])
                mean[i] = -cfg["bias_mean_shrink"] * errs.mean() / 100.0
                sd[i] = max(float(np.sqrt((errs ** 2).mean())) / 100.0, cfg["bias_sd_floor"])
            elif p == "ON" and agg.level[i] > 0:
                sd[i] = cfg["bias_sd_new_party"]
    i = PARTIES.index("ON")
    on_path = cfg.get("_root", Path(".")) / cfg["on_poll_error_file"] if cfg.get("on_poll_error_file") else None
    if agg.level[i] > 0 and on_path and on_path.exists():   # v1.12: One Nation's final-poll record in other jurisdictions
        hist = json.loads(on_path.read_text(encoding="utf-8"))["elections"]
        errs = np.array([float(e["error_pp"]) for e in hist.values()])
        mean[i] = -cfg.get("bias_mean_shrink_on", cfg["bias_mean_shrink"]) * errs.mean() / 100.0
        sd[i] = max(float(np.sqrt((errs ** 2).mean())) / 100.0, cfg.get("bias_sd_floor_on", cfg["bias_sd_floor"]))
    return grn_bt_poll_bias(mean, sd, cfg)   # Greens backtests (21 Nov, off by default): unchanged unless grn_bt_poll_bias is set


def similarity_cholesky(seats: list[Seat], cfg: dict):
    """Cholesky factor of (1-rho) I + rho K, K a Gaussian kernel on embedding distance; None without embeddings."""
    if any(s.pc is None for s in seats) or cfg["sim_rho"] <= 0:
        return None
    P = np.stack([s.pc for s in seats])
    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1)
    K = np.exp(-D2 / (2 * cfg["sim_bandwidth"] ** 2))
    S = (1 - cfg["sim_rho"]) * np.eye(len(seats)) + cfg["sim_rho"] * K
    return np.linalg.cholesky(S + 1e-9 * np.eye(len(seats)))


# ---- One Nation backtest options (21 November 2026, docs/one-nation-backtests-20261121.md). Off by default: -------------
# with neither key set this returns None and seat_primaries() is unchanged (no random numbers are used either way).
#   on_bt_pattern_elasticity k        each seat's One Nation pattern ratio p -> p ** k (k > 1: more concentrated)
#   on_bt_fed_blend w                 log p -> (1 - w) log p + w * b * log(federal 2025 ratio), b = on_bt_fed_elasticity
#   on_bt_fed_elasticity b (0.85)     South Australia 2026 state v federal 2025 One Nation, booth and district level
#   on_bt_seat_obs_weight v           replaces the weight of every seat One Nation observation (Morgan trio, Nepean and
#                                     extra_seat_on_observations; v1.23: 0.5 each), applied in seat_primaries()
# The reshaped ratios are rescaled to the original pattern's formal-weighted mean, so the statewide level is unchanged.
def on_bt_seat_patterns(seats: list[Seat], cfg: dict):
    k, w = cfg.get("on_bt_pattern_elasticity"), cfg.get("on_bt_fed_blend")
    if k is None and not w:
        return None
    base = []
    for s in seats:   # the same pattern choice as in seat_primaries()
        if s.on_pattern is not None:
            base.append(s.on_pattern)
        elif s.on_ratio_mrp is not None:
            pw = cfg["on_pattern_weights"]
            base.append(pw["federal"] * s.on_ratio + pw["embedding"] * s.on_ratio_emb + pw["mrp"] * s.on_ratio_mrp)
        else:
            base.append((1 - cfg["on_embedding_weight"]) * s.on_ratio + cfg["on_embedding_weight"] * s.on_ratio_emb)
    base = np.maximum(np.array(base, dtype=float), 1e-6)
    f = np.array([s.formal for s in seats], dtype=float)
    lp = np.log(base) * (float(k) if k is not None else 1.0)
    if w:
        fed = np.maximum(np.array([s.on_ratio for s in seats], dtype=float), 1e-6)
        lp = (1 - float(w)) * lp + float(w) * float(cfg.get("on_bt_fed_elasticity", 0.85)) * np.log(fed)
    new = np.exp(lp)
    return new * float((base * f).sum() / (new * f).sum())
# ---- end of One Nation backtest options ----------------------------------------------------------------------------------


def seat_primaries(seats: list[Seat], state: np.ndarray, state22: dict, cfg: dict,
                   rng: np.random.Generator) -> np.ndarray:
    """Return array [draws, seats, GROUPS] of seat primary shares."""
    D, S, G = state.shape[0], len(seats), len(GROUPS)
    gi = {g: i for i, g in enumerate(GROUPS)}
    pi = {p: i for i, p in enumerate(PARTIES)}
    regions = sorted(set(s.region for s in seats))
    ri = {r: i for i, r in enumerate(regions)}
    reg_sd, seat_sd = swing_scales(cfg)
    reg_eff = rng.normal(0, 1, (D, len(regions), len(PARTIES))) * reg_sd
    reg_on = rng.normal(0, cfg["on_regional_log_sd"], (D, len(regions)))
    seat_eff = rng.normal(0, 1, (D, S, len(PARTIES))) * seat_sd
    if cfg.get("seat_t_df"):   # 17 Nov experiment: Student-t seat innovations (same variance), own random stream
        df_s = float(cfg["seat_t_df"])
        trng = np.random.default_rng([int(cfg.get("seed", 0)), 121])
        seat_eff = seat_eff * np.sqrt((df_s - 2) / trng.chisquare(df_s, (D, S, 1)))
    retire =rng.normal(cfg.get("retirement_effect", 0.0), cfg.get("retirement_sd", 0.0), (D, S)).clip(0, None)
    soph = rng.normal(cfg.get("sophomore_effect", 0.0), cfg.get("sophomore_sd", 0.0), (D, S)).clip(0, None)
    fed_coef = rng.normal(cfg.get("federal_swing_coef", 0.0), cfg.get("federal_swing_coef_sd", 0.0), D)
    have_mrp = all(s.on_ratio_mrp is not None for s in seats)
    have_stack = all(s.on_pattern is not None for s in seats)
    on_sd = cfg["on_seat_log_sd_with_mrp"] if have_mrp else cfg["on_seat_log_sd"]
    if have_stack:
        on_sd = np.array([max(float(s.on_pattern_log_sd), cfg["on_pattern_log_sd_floor"]) for s in seats])
    seat_on = rng.normal(0, 1, (D, S)) * on_sd
    Lc = similarity_cholesky(seats, cfg)
    if Lc is not None:   # similar districts share part of their innovations
        seat_eff = np.einsum("ij,djp->dip", Lc, seat_eff)
        seat_on = seat_on @ Lc.T
    if any(cfg.get(k) for k in ("seat_sd_country_mult", "seat_sd_melbourne_mult", "seat_sd_open_mult")):   # 17 Nov round 2
        mult = np.array([(float(cfg.get("seat_sd_melbourne_mult") or 1.0) if is_melbourne(s) else float(cfg.get("seat_sd_country_mult") or 1.0))
                         * (float(cfg.get("seat_sd_open_mult") or 1.0) if s.retiring_party else 1.0) for s in seats])
        seat_eff = seat_eff * mult[None, :, None]
    ind_ret = rng.normal(0, cfg["ind_retention_log_sd"], (D, S))
    # 18 Nov round 3 experiments, both off by default and on their own random stream.
    seat_log = None   # per-seat addition to the leading new independent's log median (e.g. the federal independent vote)
    if cfg.get("new_ind_seat_log_file"):
        tab = pd.read_csv(Path(__file__).resolve().parents[2] / cfg["new_ind_seat_log_file"]).set_index("district_id")
        col = tab[cfg.get("new_ind_seat_log_col", "newind_fed")].astype(float)
        seat_log = {k: float(v) for k, v in col.items() if np.isfinite(v)}
    # ---- credible-independent options (22 Nov 2026, scripts/backtest_credible_ind_20261122.py); all off by default ----
    # cred_ind_file / cred_ind_col: per-seat addition to a FLAGGED new independent's log median (rows present = flagged);
    # cred_ind_extra_sd: extra log sd for flagged independents (uncertainty about the boost); cred_ind_source: where the
    # boost's extra votes come from, e.g. {"ALP": 1, "GRN": 1, "OTH": 1} (None: every party in proportion, as before);
    # cred_ind_source_mode: "share" (weight x the group's share) or "fixed" (the weights are the composition).
    # No random numbers of their own: the flagged independent reuses the seat's independent draw.
    cred = None
    if cfg.get("cred_ind_file"):
        ctab = pd.read_csv(Path(__file__).resolve().parents[2] / cfg["cred_ind_file"]).set_index("district_id")
        cred = {d: float(v) for d, v in ctab[cfg.get("cred_ind_col", "boost")].astype(float).items() if np.isfinite(v)}
    exch = None      # Labor-Greens exchange in seats where the Greens' base share is at least alp_grn_exchange_min_grn
    if cfg.get("alp_grn_exchange_sd"):
        exch = np.random.default_rng([int(cfg.get("seed", 0)), 123]).normal(0, float(cfg["alp_grn_exchange_sd"]), (D, S))
    out = np.zeros((D, S, G))
    # statewide non-ON composition in 2022 and 2026 (ALP, LNP, GRN, OTH incl IND)
    s22 = np.array([state22["ALP"], state22["LIB"] + state22["NAT"], state22["GRN"], state22["IND"] + state22["OTH"]])
    s22 = s22 / s22.sum()
    s26 = state[:, [pi["ALP"], pi["LNP"], pi["GRN"], pi["OTH"]]]
    s26 = s26 / s26.sum(axis=1, keepdims=True)
    if cfg.get("swing_amplify"):   # 17 Nov round 2: polls understate the size of the swing from the last election
        s26 = np.clip(s22[None, :] + float(cfg["swing_amplify"]) * (s26 - s22[None, :]), 1e-3, None)
        s26 = s26 / s26.sum(axis=1, keepdims=True)
    swing = s26 - s22   # [D, 4] additive
    lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
    lswing = lg(s26) - lg(s22)[None, :]   # [D, 4] log-odds
    on_bt_pat = on_bt_seat_patterns(seats, cfg)   # One Nation backtest options (21 Nov 2026); None when off
    for k, seat in enumerate(seats):
        r = ri[seat.region]
        if seat.on_pattern is not None:
            pattern = seat.on_pattern
        elif seat.on_ratio_mrp is not None:
            w = cfg["on_pattern_weights"]
            pattern = w["federal"] * seat.on_ratio + w["embedding"] * seat.on_ratio_emb + w["mrp"] * seat.on_ratio_mrp
        else:
            pattern = (1 - cfg["on_embedding_weight"]) * seat.on_ratio + cfg["on_embedding_weight"] * seat.on_ratio_emb
        if on_bt_pat is not None:   # One Nation backtest options (21 Nov 2026)
            pattern = float(on_bt_pat[k])
        ratio = (1 - cfg["on_pattern_shrink"]) + cfg["on_pattern_shrink"] * pattern
        if cfg.get("use_morgan_seat_poll", True):
            for obs in SEAT_ON_OBSERVATIONS + list(cfg.get("extra_seat_on_observations") or []):
                if seat.district_id in obs["districts"]:
                    ow = obs["weight"] if cfg.get("on_bt_seat_obs_weight") is None else float(cfg["on_bt_seat_obs_weight"])   # ON backtest option
                    ratio = (1 - ow) * ratio + ow * obs["on_share"] / obs["state_on_at_time"]
        on_sd_k = float(on_sd[k]) if isinstance(on_sd, np.ndarray) else on_sd
        on = state[:, pi["ON"]] * ratio * np.exp(reg_on[:, r] + seat_on[:, k] - 0.5 * (cfg["on_regional_log_sd"] ** 2 + on_sd_k ** 2))
        on = np.clip(on, cfg["on_floor"], 0.60) if seat.stands["ON"] else np.zeros(D)
        sh = dict(seat.base22 or seat.share22)
        if seat.top_base and sh.get(seat.top_base, 0.0) > 0:   # v1.9: strip the anchor election's donkey gain
            sh = ballot_top_remove(sh, seat.top_base, ballot_gain(seat.top_base, cfg))
        # v1.20: a returning independent starts from that candidate's own vote, not the sum of every independent in the seat
        ind22 = seat.ind_top22 if cfg.get("ind_base_top_only") else sh["IND"]
        if seat.ind_departed > 0:   # v1.8: a departed independent's vote mostly returns to the parties
            seat_ind_departed = min(seat.ind_departed, sh["IND"])
            moved = seat_ind_departed * cfg.get("departed_ind_to_parties", 0.0)
            major = sh["ALP"] + sh["LIB"] + sh["NAT"] + sh["GRN"]
            for g in ("ALP", "LIB", "NAT", "GRN"):
                sh[g] += moved * sh[g] / major
            sh["IND"] -= seat_ind_departed
            sh["OTH"] += seat_ind_departed - moved
        base_on22 = sh["ON"]
        rem22 = max(1.0 - base_on22, 1e-6)
        b_alp = sh["ALP"] / rem22
        b_lnp = (sh["LIB"] + sh["NAT"]) / rem22
        b_grn = sh["GRN"] / rem22
        b_oth = (sh["OTH"] + sh["IND"]) / rem22
        base = np.array([b_alp, b_lnp, b_grn, b_oth])
        scale = cfg.get("swing_scale", "additive")
        if scale in ("logit", "mixed"):
            z = lg(base)[None, :] + lswing
            comp_l = 1 / (1 + np.exp(-z))
            comp_l = comp_l / comp_l.sum(axis=1, keepdims=True)   # proportional swing, renormalised within the non-ON composition
            comp_a = base[None, :] + swing
            comp = comp_l if scale == "logit" else 0.5 * comp_l + 0.5 * comp_a
            if cfg.get("swing_scale_party"):   # 17 Nov experiment: per-party shape, e.g. Greens proportional, majors uniform
                cols = ("ALP", "LNP", "GRN", "OTH")
                comp_af = comp_a
                if cfg.get("swing_additive_floor_frac"):   # a party keeps at least this share of its seat base under uniform swing
                    comp_af = np.maximum(comp_a, float(cfg["swing_additive_floor_frac"]) * base[None, :])
                pick = {"logit": comp_l, "additive": comp_af, "mixed": 0.5 * comp_l + 0.5 * comp_a, "mixed75": 0.25 * comp_l + 0.75 * comp_af}
                comp = np.stack([pick[cfg["swing_scale_party"].get(c, scale)][:, j] for j, c in enumerate(cols)], axis=1)
                comp = np.clip(comp, 1e-4, None)
                comp = comp / comp.sum(axis=1, keepdims=True)
            comp = comp + reg_eff[:, r, :][:, [0, 1, 2, 4]] + seat_eff[:, k, :][:, [0, 1, 2, 4]]
        else:
            comp = base[None, :] + swing + reg_eff[:, r, :][:, [0, 1, 2, 4]] + seat_eff[:, k, :][:, [0, 1, 2, 4]]
        if seat.fed_rel:   # v1.10: the district's federal relative 2PP swing, partly carried into the state Labor-v-Coalition split
            shift = fed_coef * seat.fed_rel
            comp[:, 0] = comp[:, 0] + shift
            comp[:, 1] = comp[:, 1] - shift
        pv = float(cfg.get("personal_vote_country_mult", 1.0)) if not is_melbourne(seat) else float(cfg.get("personal_vote_melbourne_mult", 1.0))
        if seat.retiring_party in ("ALP", "LNP", "GRN"):
            j = {"ALP": 0, "LNP": 1, "GRN": 2}[seat.retiring_party]
            comp[:, j] = comp[:, j] - pv * retire[:, k]
        if seat.sophomore_party in ("ALP", "LNP", "GRN"):   # v1.10: also when the same party lost its 2022 member (by-election seats)
            j = {"ALP": 0, "LNP": 1, "GRN": 2}[seat.sophomore_party]
            comp[:, j] = comp[:, j] + pv * soph[:, k]
        if exch is not None and b_grn >= float(cfg.get("alp_grn_exchange_min_grn", 0.15)):
            comp[:, 0] = comp[:, 0] + exch[:, k]
            comp[:, 2] = comp[:, 2] - exch[:, k]
        tilt = float(cfg.get("on_alp_tilt", 0.0))
        if tilt and seat.stands["ON"]:   # sensitivity: where One Nation runs above its state level, Labor loses extra share of the rest
            comp[:, 0] = comp[:, 0] - tilt * (on - state[:, pi["ON"]])
        comp = np.clip(comp, 0.002, None)
        # independent handling inside the "other" slice
        ind = np.zeros(D)
        ind_extra = None   # credible-independent votes still to be taken from named groups (cred_ind_source)
        if seat.ind_mode == "recontest":
            mu = cfg.get("ind_retention_log_mean")   # v1.20: measured mean of log(share / previous share); None = mean-preserving
            ind = ind22 * np.exp(ind_ret[:, k] + (-0.5 * cfg["ind_retention_log_sd"] ** 2 if mu is None else float(mu)))
        elif seat.ind_mode == "new":
            if cfg.get("new_ind_log_sd"):   # v1.20: the leading new independent's share, lognormal about a seat median
                med = np.log(cfg["new_ind_share"]) + cfg.get("new_ind_extra_log", 0.0) * max(seat.n_new_ind - 1, 0)                     + (0.0 if is_melbourne(seat) else cfg.get("new_ind_country_log", 0.0))
                if cfg.get("new_ind_margin_log") or cfg.get("new_ind_open_log"):   # 17 Nov round 2: safe and open seats
                    two = sorted([seat.share22["ALP"], seat.share22["LIB"] + seat.share22["NAT"], seat.share22["GRN"], seat.share22["IND"]])[-2:]
                    med = med + float(cfg.get("new_ind_margin_log") or 0.0) * ((two[1] - two[0]) / 2 - 0.10)                         + (float(cfg.get("new_ind_open_log") or 0.0) if seat.retiring_party else 0.0)
                if seat_log is not None:
                    med = med + seat_log.get(seat.district_id, 0.0)
                ind = np.clip(np.exp(med + ind_ret[:, k] / cfg["ind_retention_log_sd"] * cfg["new_ind_log_sd"]), 0.0, 0.6)
                if cred is not None and seat.district_id in cred:   # credible independent (off by default)
                    sd_c = float(np.hypot(cfg["new_ind_log_sd"], float(cfg.get("cred_ind_extra_sd") or 0.0)))
                    ind_c = np.clip(np.exp(med + cred[seat.district_id] + ind_ret[:, k] / cfg["ind_retention_log_sd"] * sd_c), 0.0, 0.6)
                    if cfg.get("cred_ind_source"):
                        ind_extra = np.maximum(ind_c - ind, 0.0)   # moved in after the composition is set, from named groups
                        ind = np.minimum(ind, ind_c)
                    else:
                        ind = ind_c
            else:
                ind = np.full(D, cfg["new_ind_share"]) * np.exp(ind_ret[:, k])
        oth = np.clip(comp[:, 3] - ind, 0.01, None)
        # Coalition split
        lib22, nat22 = seat.share22["LIB"], seat.share22["NAT"]
        if seat.stands["LIB"] and seat.stands["NAT"]:
            f = lib22 / (lib22 + nat22) if (lib22 + nat22) > 0 else 0.6
            if lib22 == 0 or nat22 == 0:  # new three-cornered contest: 2022 party keeps most
                f = 0.7 if lib22 > 0 else 0.3
            lib, nat = comp[:, 1] * f, comp[:, 1] * (1 - f)
        elif seat.stands["NAT"]:
            lib, nat = np.zeros(D), comp[:, 1]
        else:
            lib, nat = comp[:, 1], np.zeros(D)
        alp_v = comp[:, 0] if seat.stands.get("ALP", True) else np.zeros(D)
        grn_v = comp[:, 2] if seat.stands.get("GRN", True) else np.zeros(D)
        if not (seat.stands["LIB"] or seat.stands["NAT"]):
            lib, nat = np.zeros(D), np.zeros(D)
        non_on = np.stack([alp_v, lib, nat, grn_v, ind, oth], axis=1)  # ALP LIB NAT GRN IND OTH
        non_on = non_on / non_on.sum(axis=1, keepdims=True) * (1 - on)[:, None]
        if ind_extra is not None:   # credible independent: its extra votes come from the named groups only
            src, cols = cfg["cred_ind_source"], {"ALP": [0], "LNP": [1, 2], "GRN": [3], "OTH": [5]}
            want = ind_extra * (1 - on)
            have = {g: non_on[:, c].sum(axis=1) for g, c in cols.items() if float(src.get(g, 0.0)) > 0}
            if cfg.get("cred_ind_source_mode", "share") == "fixed":
                wts = {g: np.full(D, float(src[g])) for g in have}
            else:
                wts = {g: float(src[g]) * have[g] for g in have}
            tot = np.maximum(sum(wts.values()), 1e-12)
            taken = np.zeros(D)
            for g, c in cols.items():
                if g not in have:
                    continue
                t = np.minimum(want * wts[g] / tot, 0.9 * have[g])
                part = non_on[:, c] / np.maximum(have[g], 1e-12)[:, None]
                non_on[:, c] = non_on[:, c] - t[:, None] * part
                taken = taken + t
            non_on[:, 4] = non_on[:, 4] + taken
        out[:, k, gi["ALP"]] = non_on[:, 0]
        out[:, k, gi["LIB"]] = non_on[:, 1]
        out[:, k, gi["NAT"]] = non_on[:, 2]
        out[:, k, gi["GRN"]] = non_on[:, 3]
        out[:, k, gi["IND"]] = non_on[:, 4]
        out[:, k, gi["OTH"]] = non_on[:, 5]
        out[:, k, gi["ON"]] = on
        if seat.top_now:   # v1.9: the 2026 top-of-ballot candidate's donkey gain, expected over an unpublished draw
            out[:, k, :] = ballot_top_add(out[:, k, :], seat.top_now, gi, cfg)
    return out


REGION_SPLIT_GROUPS = {"ALP": ["ALP"], "LNP": ["LIB", "NAT"], "GRN": ["GRN"], "ON": ["ON"]}


def region_split_targets(path: Path, as_of: date, cfg: dict) -> dict:
    """Pooled country/Melbourne ratio per party from published poll crosstabs (data/curated/forecast-v1/
    region-split-polls-2026.csv). Each poll gives log(country share / Melbourne share); Melbourne is the mean of inner and
    outer areas where a pollster splits them. Variance from subsample sizes (design effect, country fraction 0.29),
    inflated by 2^(age / half-life) for older polls; pooled by inverse variance with a definitional floor."""
    df = pd.read_csv(path)
    deff = cfg.get("design_effect", 1.5)
    out = {}
    for party in cfg.get("region_split_parties", list(REGION_SPLIT_GROUPS)):
        col = party.lower()
        logs, varis = [], []
        for (pollster, end), g in df.groupby(["pollster", "fieldwork_end"]):
            country = g[g.area.str.contains("Country|Regional", case=False)]
            melb = g[~g.area.str.contains("Country|Regional", case=False)]
            if country.empty or melb.empty:
                continue
            n = float(g["sample"].iloc[0])
            pc, pm = float(country[col].mean()) / 100.0, float(melb[col].mean()) / 100.0
            nc, nm = n * 0.29 / deff, n * 0.71 / deff
            v = (1 - pc) / (pc * nc) + (1 - pm) / (pm * nm)
            age = max((as_of - _as_date(end)).days, 0)
            v *= 2.0 ** (age / cfg.get("region_split_half_life_days", 90.0))
            logs.append(math.log(pc / pm))
            varis.append(v)
        if not logs:
            continue
        wts = 1.0 / np.array(varis)
        mean = float((np.array(logs) * wts).sum() / wts.sum())
        se = math.sqrt(1.0 / wts.sum() + cfg.get("region_split_definition_sd", 0.03) ** 2)
        out[party] = {"log_ratio": mean, "log_se": se, "ratio": math.exp(mean), "n_polls": len(logs)}
    return out


def region_split_calibration(prim: np.ndarray, seats: list[Seat], cfg: dict, as_of: date,
                             rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """v1.12: pull each party's country/Melbourne balance toward the pooled poll crosstabs. Per draw the new log ratio is
    a precision-weighted combination of the model's own ratio and a draw from the poll estimate (weights from the model's
    across-draw variance and the poll standard error), so the result has the posterior mean and variance. Party
    statewide totals are preserved by scaling Melbourne and country seats with offsetting factors; each seat is then
    renormalised. Within-region differences between seats are untouched.

    v1.15: for parties in cfg["region_split_rural_only_parties"] the provincial cities take Melbourne's factor and the
    rural seats carry the whole country correction. The target ratio, its weight, the poll draw, the Melbourne factor
    and the country total are the same as with one country factor, so only the split inside the country changes."""
    path = cfg.get("_root", Path(".")) / cfg["region_split_file"] if cfg.get("region_split_file") else None
    if not path or not path.exists():
        return prim, {}
    targets = region_split_targets(path, as_of, cfg)
    melb = np.array([is_melbourne(s) for s in seats])
    prov = np.array([s.district_id in PROVINCIAL_CITY_DISTRICTS for s in seats]) & ~melb
    rur = ~melb & ~prov
    rural_only = set(cfg.get("region_split_rural_only_parties") or [])
    wts = np.array([s.formal for s in seats], dtype=float)
    a = wts[~melb].sum() / wts.sum()   # country share of the statewide weight
    gi = {g: i for i, g in enumerate(GROUPS)}
    out = prim.copy()
    report = {"melbourne_seats": int(melb.sum()), "country_seats": int((~melb).sum()), "country_weight_share": float(a),
              "parties": {}}
    if rural_only:
        report.update({"rural_only_parties": sorted(rural_only), "provincial_city_seats": int(prov.sum()),
                       "rural_seats": int(rur.sum())})
    for party, t in targets.items():
        idx = [gi[g] for g in REGION_SPLIT_GROUPS[party]]
        tot = out[:, :, idx].sum(axis=2)
        s_m = (tot[:, melb] * wts[melb]).sum(axis=1) / wts[melb].sum()
        s_c = (tot[:, ~melb] * wts[~melb]).sum(axis=1) / wts[~melb].sum()
        ok = (s_m > 0) & (s_c > 0)
        cur = np.where(ok, np.log(np.where(ok, s_c, 1.0) / np.where(ok, s_m, 1.0)), 0.0)
        v_model = float(cur[ok].var()) if ok.any() else 0.0
        w = v_model / (v_model + t["log_se"] ** 2) if v_model > 0 else 0.0
        poll = rng.normal(t["log_ratio"], t["log_se"], len(cur))
        new = (1 - w) * cur + w * poll
        k = np.exp(new - cur)   # f_country / f_melbourne
        f_m = (a * s_c + (1 - a) * s_m) / np.where(ok, a * k * s_c + (1 - a) * s_m, 1.0)
        f_m = np.where(ok, f_m, 1.0)
        f_c = np.where(ok, k * f_m, 1.0)
        extra = {}
        if party in rural_only and prov.any() and rur.any():
            # keep the country total f_c * s_c: c_p * f_m * s_p + c_r * f_r * s_r = f_c * s_c (c = shares of country weight)
            c_p, c_r = wts[prov].sum() / wts[~melb].sum(), wts[rur].sum() / wts[~melb].sum()
            s_p = (tot[:, prov] * wts[prov]).sum(axis=1) / wts[prov].sum()
            s_r = (tot[:, rur] * wts[rur]).sum(axis=1) / wts[rur].sum()
            f_r = (f_c * s_c - c_p * f_m * s_p) / np.where(s_r > 0, c_r * s_r, 1.0)
            clipped = ok & (s_r > 0) & (f_r < 0.05)
            f_r = np.where(ok & (s_r > 0), np.maximum(f_r, 0.05), 1.0)
            for j in idx:
                out[:, melb, j] = out[:, melb, j] * f_m[:, None]
                out[:, prov, j] = out[:, prov, j] * f_m[:, None]
                out[:, rur, j] = out[:, rur, j] * f_r[:, None]
            extra = {"rural_only": True, "provincial_factor_mean": float(f_m.mean()), "rural_factor_mean": float(f_r.mean()),
                     "rural_factor_clipped_draws": int(clipped.sum())}
        else:
            for j in idx:
                out[:, melb, j] = out[:, melb, j] * f_m[:, None]
                out[:, ~melb, j] = out[:, ~melb, j] * f_c[:, None]
        report["parties"][party] = {"poll_ratio": t["ratio"], "poll_log_se": t["log_se"], "n_polls": t["n_polls"],
                                    "model_ratio_before": float(np.exp(cur[ok].mean())), "weight_on_polls": float(w),
                                    "ratio_after": float(np.exp(new[ok].mean())),
                                    "melbourne_factor_mean": float(f_m.mean()), "country_factor_mean": float(f_c.mean()),
                                    **extra}
    out = out / out.sum(axis=2, keepdims=True)
    return out, report


def ballot_gain(g: str, cfg: dict) -> float:
    return cfg.get("ballot_top_gain_minor", 0.0) if g in ("IND", "OTH") else cfg.get("ballot_top_gain", 0.0)


def ballot_top_remove(sh: dict, g: str, gain: float) -> dict:
    """Remove a top-of-ballot gain from group g, returning it to the other groups in proportion."""
    gain = min(gain, 0.8 * sh[g])
    rest = 1.0 - sh[g]
    if gain <= 0 or rest <= 0:
        return sh
    return {k: (v - gain if k == g else v * (rest + gain) / rest) for k, v in sh.items()}


def ballot_top_add(x: np.ndarray, top: dict, gi: dict, cfg: dict) -> np.ndarray:
    """x [draws, groups] seat primaries; add each possible top group's gain (taken from the others in proportion),
    weighted by its probability of holding the top position. Unassigned probability leaves x unchanged."""
    out = x * (1.0 - sum(top.values()))
    for g, p in top.items():
        s = x[:, gi[g]]
        gain = np.minimum(ballot_gain(g, cfg), 0.95 * (1.0 - s))
        if p <= 0 or not np.all(s > 0):
            out = out + p * x
            continue
        y = x * ((1.0 - s - gain) / (1.0 - s))[:, None]
        y[:, gi[g]] = s + gain
        out = out + p * y
    return out


def _coalition(g: str) -> str:
    return "LNP" if g in ("LIB", "NAT") else g


_EMPIRICAL_FLOWS: dict = {}


def load_empirical_flows(path: Path, cfg: dict) -> dict:
    """(first, second) -> origin -> (rate_to_first, concentration) from Victorian ledgers."""
    out = {}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for r in df.itertuples():
        conc = r.beta_concentration
        if not isinstance(conc, (int, float)) or conc != conc or r.n_districts < 2:
            conc = 15.0
        conc = float(min(max(conc, cfg["flow_min_conc"]), cfg["flow_max_conc"]))
        out.setdefault((r.first, r.second), {})[r.origin] = (float(r.rate_to_first), conc)
    return out


def coalition_on_rates(cfg: dict, recommend: bool | None):
    """v1.12: final-pair table and transfer weights under the drawn Coalition preference decision. Returns the module
    tables unchanged when the scenario is off, so the number and order of random draws never changes."""
    if recommend is None or cfg.get("coalition_recommends_on_p") is None:
        return FINAL_PAIR_RATES, TRANSFER_WEIGHTS
    m_rec, m_open = cfg["lnp_to_alp_vs_on_recommended"], cfg["lnp_to_alp_vs_on_open"]
    m = m_rec if recommend else m_open
    rates = {k: dict(v) for k, v in FINAL_PAIR_RATES.items()}
    rates[("ALP", "ON")]["LNP"] = m
    tw = {o: dict(ws) for o, ws in TRANSFER_WEIGHTS.items()}
    for o, share in (("LIB", m), ("NAT", m_rec)):   # Nationals resolved to recommend One Nation above Labor
        tot = tw[o]["ALP"] + tw[o]["ON"]
        tw[o]["ALP"], tw[o]["ON"] = tot * share, tot * (1 - share)
    return rates, tw


def liberal_order_assumption(cfg: dict) -> str:
    """Assumption text for the Liberal order of Labor versus the Greens: a shared draw while the order is only reported
    (v1.13), or fixed once the registered Liberal cards show it (v1.14, p >= 1)."""
    opens = ", ".join(s.replace("-", " ").title() for s in sorted(cfg.get("lib_open_alp_grn_seats") or []))
    open_txt = f"; {cfg['lnp_to_alp_vs_grn_open']:.0%} where the Liberals run no directed card ({opens})." if opens else "."
    if float(cfg["lib_alp_over_grn_p"]) >= 1.0:
        cards = cfg.get("lib_cards_registered") or {}
        seen = (f" ({cards['seats_with_cards']} seats by {cards['as_of']})"
                if cards.get("seats_with_cards") and cards.get("as_of") else "")
        return (f"Liberal order of Labor versus the Greens: the registered Liberal cards{seen} put Labor above the Greens in "
                f"every seat with a card, so {cfg['lnp_to_alp_vs_grn_directed']:.0%} of Liberal preferences reach Labor in "
                f"Labor-v-Greens counts (Victoria 2014 and 2018, when the card said so; "
                f"{cfg['lnp_to_alp_vs_grn_reversed']:.0%} in 2022, when it favoured the Greens)" + open_txt)
    return (f"Liberal order of Labor versus the Greens: one shared draw per simulated election; with probability "
            f"{cfg['lib_alp_over_grn_p']:.2f} Liberal cards put Labor above the Greens (reported, not yet registered) and "
            f"{cfg['lnp_to_alp_vs_grn_directed']:.0%} of Liberal preferences reach Labor in Labor-v-Greens counts (Victoria 2014 and 2018), "
            f"otherwise {cfg['lnp_to_alp_vs_grn_reversed']:.0%} (2022, Greens above Labor)" + open_txt)


def lib_alp_grn_mean(cfg: dict, lib_first: bool | None):
    """v1.13: Liberal-ballot rate to Labor in Labor-v-Greens counts under the drawn Liberal card order, or None (off)."""
    if lib_first is None or cfg.get("lib_alp_over_grn_p") is None:
        return None
    return cfg["lnp_to_alp_vs_grn_directed"] if lib_first else cfg["lnp_to_alp_vs_grn_reversed"]


def draw_flows(cfg: dict, rng: np.random.Generator, recommend: bool | None = None, lib_first: bool | None = None):
    """Per-election Beta draws of final-pair rates; transfer weights get mild lognormal noise.
    Victorian empirical rates (flows_file) override the declared table where observed. ``recommend`` (v1.12) is the
    simulated Coalition decision on One Nation versus Labor; None keeps the declared table. ``lib_first`` (v1.13) is the
    simulated Liberal card order of Labor above the Greens; it replaces the pooled 2014-2022 Liberal rate in
    Labor-v-Greens counts with the rate under that order, using the same single random draw."""
    c = cfg["flow_concentration"]
    emp = _EMPIRICAL_FLOWS.get("table", {})
    final_rates, transfer = coalition_on_rates(cfg, recommend)
    final_rates, transfer = grn_bt_flow_tables(cfg, final_rates, transfer)   # Greens backtests (21 Nov, off by default)
    m_lib = lib_alp_grn_mean(cfg, lib_first)
    if m_lib is not None:
        transfer = {o: dict(ws) for o, ws in transfer.items()}
        tot = transfer["LIB"]["ALP"] + transfer["LIB"]["GRN"]
        transfer["LIB"]["ALP"], transfer["LIB"]["GRN"] = tot * m_lib, tot * (1 - m_lib)

    def scen(pair, o, m, k):
        if m_lib is not None and pair == ("ALP", "GRN") and o == "LNP":
            return m_lib, cfg["lnp_alp_grn_concentration"]
        return m, k

    pair_rates = {}
    for pair, rates in final_rates.items():
        pair_rates[pair] = {}
        for o, p in rates.items():
            m, k = scen(pair, o, p, c)
            pair_rates[pair][o] = rng.beta(m * k, (1 - m) * k)
    for pair, origins in emp.items():
        rev = (pair[1], pair[0])
        for o, (m, k) in origins.items():
            m, k = scen(pair, o, m, k)
            if pair in pair_rates:
                pair_rates[pair][o] = rng.beta(m * k, (1 - m) * k)
            elif rev in pair_rates:
                pair_rates[rev][o] = 1 - rng.beta(m * k, (1 - m) * k)
            else:
                pair_rates.setdefault(pair, {})[o] = rng.beta(m * k, (1 - m) * k)
    for key, origins in (cfg.get("final_pair_rates_override") or {}).items():   # Tally Room idea 1 (off by default)
        a, b = key.split("-")
        k = float(cfg.get("final_pair_override_concentration") or c)
        for o, m in origins.items():
            x = rng.beta(float(m) * k, (1 - float(m)) * k)
            if (a, b) in pair_rates or (b, a) not in pair_rates:
                pair_rates.setdefault((a, b), {})[o] = x
            else:
                pair_rates[(b, a)][o] = 1 - x
    weights ={o: {d: w * math.exp(rng.normal(0, 0.15)) for d, w in ws.items()} for o, ws in transfer.items()}
    return pair_rates, weights


def final_pair_rate(pair_rates, a: str, b: str, origin: str, seat_noise: float, adjust: dict | None = None) -> float:
    """Probability that ``origin`` ballots prefer candidate group a over b. ``adjust`` maps
    (origin, first, second) -> additive shift to the rate for ``first`` (seat-specific flow lean)."""
    ca, cb, co = _coalition(a), _coalition(b), _coalition(origin)
    if co == ca:
        return 1.0
    if co == cb:
        return 0.0
    # Noise is a shift toward the FIRST group in the stored pair. Apply and
    # bound it once in that reference direction, then complement on reversal.
    # Otherwise querying both directions adds the same noise twice and an
    # untied count can change winner just by reordering its input dictionary.
    key = (ca, cb)
    if key in pair_rates:
        flip = False
    elif (cb, ca) in pair_rates:
        key, flip = (cb, ca), True
    else:
        key = tuple(sorted((ca, cb)))
        flip = (ca, cb) != key if ca != cb else a > b
    p = pair_rates.get(key, {}).get(co, 0.5)
    if adjust:
        p += adjust.get((co, key[0], key[1]), 0.0) - adjust.get((co, key[1], key[0]), 0.0)
    p = float(min(max(p + seat_noise, 0.02), 0.98))
    return 1 - p if flip else p


def irv(shares: dict, pair_rates, weights, seat_noise: float, adjust: dict | None = None, keep: tuple | None = None):
    """Aggregate IRV with origin-tracked parcels. ``shares``: group -> first-preference share (>0 only).

    Returns (winner_group, final_pair_sorted, final_share_of_first_in_pair).
    OTH is a lump of minor candidates: it is always excluded first.
    ``keep`` names two groups that are never excluded, which reproduces a two-candidate-preferred count for that pair
    (optional probe; the default count is unchanged).
    """
    tally = {g: {g: v} for g, v in shares.items() if v > 0}   # holder -> origin -> votes
    if len(tally) == 1:
        g = next(iter(tally))
        return g, (g, g), 1.0
    while True:
        totals = {h: sum(o.values()) for h, o in tally.items()}
        if len(totals) == 2:
            (a, ta), (b, tb) = sorted(totals.items(), key=lambda kv: -kv[1])
            pair = tuple(sorted((a, b)))
            return a, pair, totals[pair[0]] / (ta + tb)
        if "OTH" in totals and not (keep and "OTH" in keep):
            loser = "OTH"
        elif keep:
            loser = min((h for h in totals if h not in keep), key=lambda h: totals[h])
        else:
            loser = min(totals, key=lambda h: totals[h])
        parcels = tally.pop(loser)
        remaining = list(tally)
        for origin, v in parcels.items():
            if len(remaining) == 2:
                a, b = remaining
                p = final_pair_rate(pair_rates, a, b, origin, seat_noise, adjust)
                tally[a][origin] = tally[a].get(origin, 0.0) + v * p
                tally[b][origin] = tally[b].get(origin, 0.0) + v * (1 - p)
            else:
                w = np.array([weights[origin].get(h, 0.3) if origin != h else 1.0 for h in remaining])
                w = w / w.sum()
                for h, wi in zip(remaining, w):
                    tally[h][origin] = tally[h].get(origin, 0.0) + v * wi


def two_party(shares: dict, pair_rates, seat_noise: float, adjust: dict | None = None):
    """Labor versus Coalition two-party-preferred share for Labor, or None if either side is absent."""
    lnp = shares.get("LIB", 0) + shares.get("NAT", 0)
    if shares.get("ALP", 0) <= 0 or lnp <= 0:
        return None
    alp = shares["ALP"]
    for g, v in shares.items():
        if g in ("ALP", "LIB", "NAT") or v <= 0:
            continue
        alp += v * final_pair_rate(pair_rates, "ALP", "LNP", g, seat_noise, adjust)
    return float(min(max(alp, 0.0), 1.0))


def simulate(seats: list[Seat], prim: np.ndarray, cfg: dict, rng: np.random.Generator,
             recommend: np.ndarray | None = None, lib_first: np.ndarray | None = None, pair_probe: dict | None = None):
    """``recommend`` (v1.12): per-draw Coalition decision to put One Nation above Labor, or None (declared table).
    ``lib_first`` (v1.13): per-draw Liberal card order of Labor above the Greens, or None (pooled 2014-2022 rate).
    ``pair_probe`` (optional): seat index -> (shares, (a, b)). For those seats each draw's flows also count
    the given shares down to the pair a, b; the result is returned as a fifth array (share of the first of the sorted
    pair). It uses no random numbers, so the other four arrays are the same with or without it."""
    D, S, _ = prim.shape
    winners = np.empty((D, S), dtype=object)
    pairs = np.empty((D, S), dtype=object)
    pair_share = np.full((D, S), np.nan)
    tpp = np.full((D, S), np.nan)
    # seat-specific One Nation flow lean: ON voters in Coalition-leaning seats flow more strongly to the Coalition
    slope = cfg.get("on_flow_lean_slope", 0.0)
    lnp22 = np.array([s.share22["LIB"] + s.share22["NAT"] for s in seats])
    formal = np.array([s.formal for s in seats], dtype=float)
    lean = lnp22 - (lnp22 * formal).sum() / formal.sum()
    adjusts = [{("ON", "LNP", "ALP"): float(slope * lean[k])} for k in range(S)]
    # v1.12: Nationals resolved to put One Nation above Labor; when the drawn Liberal decision is "open", NAT-only seats
    # still take the recommended Coalition rate in Labor-v-One-Nation counts
    nat_only = [s.stands.get("NAT", False) and not s.stands.get("LIB", False) for s in seats]
    nat_shift = cfg["lnp_to_alp_vs_on_recommended"] - cfg["lnp_to_alp_vs_on_open"] \
        if cfg.get("coalition_recommends_on_p") is not None else 0.0
    adjusts_open = [{**a, ("LNP", "ALP", "ON"): nat_shift} if nat_only[k] else a for k, a in enumerate(adjusts)]
    # v1.13 seat-level card evidence, as shifts from the drawn statewide regime toward each seat's own rate
    ids = [s.district_id for s in seats]
    on_targets = dict(cfg.get("seat_lnp_alp_vs_on_rate") or {})
    open_grn = set(cfg.get("lib_open_alp_grn_seats") or [])
    fixed = cfg.get("seat_flow_adjust") or {}
    v113 = bool(on_targets or open_grn or fixed)
    variants = {}

    def seat_adjusts(rec, lib):
        if not v113:
            return adjusts_open if rec is False else adjusts
        key = (rec, lib)
        if key not in variants:
            base = adjusts_open if rec is False else adjusts
            if rec is None or cfg.get("coalition_recommends_on_p") is None:
                m_on = FINAL_PAIR_RATES[("ALP", "ON")]["LNP"]
            else:
                m_on = cfg["lnp_to_alp_vs_on_recommended"] if rec else cfg["lnp_to_alp_vs_on_open"]
            m_lib = lib_alp_grn_mean(cfg, lib)
            out = []
            for k, a in enumerate(base):
                a = dict(a)
                if ids[k] in on_targets:   # replaces the regime rate (and the v1.12 Nationals rule) in this seat
                    a[("LNP", "ALP", "ON")] = float(on_targets[ids[k]]) - m_on
                if m_lib is not None and ids[k] in open_grn:
                    a[("LNP", "ALP", "GRN")] = cfg["lnp_to_alp_vs_grn_open"] - m_lib
                for origin, first, second, shift in fixed.get(ids[k], []):
                    a[(origin, first, second)] = a.get((origin, first, second), 0.0) + float(shift)
                out.append(a)
            variants[key] = out
        return variants[key]

    probe = np.full((D, S), np.nan) if pair_probe else None
    for d in range(D):
        rec = None if recommend is None else bool(recommend[d])
        lib = None if lib_first is None else bool(lib_first[d])
        pair_rates, weights = draw_flows(cfg, rng, rec, lib)
        adj = seat_adjusts(rec, lib)
        noise = rng.normal(0, cfg["seat_flow_sd"], S)
        for k, seat in enumerate(seats):
            shares = {g: float(prim[d, k, i]) for i, g in enumerate(GROUPS) if prim[d, k, i] > 0}
            w, pr, fs = irv(shares, pair_rates, weights, noise[k], adj[k])
            winners[d, k], pairs[d, k], pair_share[d, k] = w, "-".join(pr), fs
            t = two_party(shares, pair_rates, noise[k], adj[k])
            if t is not None:
                tpp[d, k] = t
            if pair_probe and k in pair_probe:
                probe[d, k] = irv(pair_probe[k][0], pair_rates, weights, noise[k], adj[k], keep=pair_probe[k][1])[2]
    if pair_probe:
        return winners, pairs, pair_share, tpp, probe
    return winners, pairs, pair_share, tpp


def statewide_primary_draws(prim: np.ndarray, formal: np.ndarray) -> np.ndarray:
    """Aggregate district primaries within each election draw using fixed anchor formal weights.

    These are transported historical volumes, not observed/predicted 2026 formals.
    Combine LIB+NAT and IND+OTH before summarising; retain the joint draw order.
    """
    prim = np.asarray(prim, dtype=float)
    formal = np.asarray(formal, dtype=float)
    if prim.ndim != 3 or prim.shape[2] != len(GROUPS) or formal.shape != (prim.shape[1],):
        raise ValueError("district primary draws and formal weights must align")
    if not np.isfinite(formal).all() or (formal <= 0).any():
        raise ValueError("positive finite anchor formal weights required for every district")
    weighted = np.einsum("dsg,s->dg", prim, formal / formal.sum())
    gi = {g: i for i, g in enumerate(GROUPS)}
    return np.column_stack([weighted[:, gi["ALP"]], weighted[:, gi["LIB"]] + weighted[:, gi["NAT"]],
                            weighted[:, gi["GRN"]], weighted[:, gi["ON"]],
                            weighted[:, gi["IND"]] + weighted[:, gi["OTH"]]])


def seat_lead_probabilities(counts: np.ndarray, categories: list[str]) -> dict:
    """Separate outright and tied seat leads without ordering or random tie breaks.

    Outright party events plus any tie partition the joint elections. Joint-lead
    events include every party tied at the maximum and can overlap.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or not len(counts) or counts.shape[1] != len(categories):
        raise ValueError("nonempty seat-count draws must align with party categories")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("finite nonnegative integer seat counts required")
    leaders = counts == counts.max(axis=1, keepdims=True)
    tied = leaders.sum(axis=1) > 1
    return {"p_most_seats": {c: float((leaders[:, j] & ~tied).mean()) for j, c in enumerate(categories)},
            "p_tied_most_seats": float(tied.mean()),
            "p_joint_most_seats": {c: float(leaders[:, j].mean()) for j, c in enumerate(categories)},
            "most_seats_definition": "p_most_seats is an outright lead; p_joint_most_seats includes ties; party outright leads plus any tie sum to one"}


def summarise(seats, state, prim, winners, pairs, pair_share, tpp, agg: PollAggregate, cfg: dict, as_of: date):
    D, S, _ = prim.shape
    seat_rows = []
    sim = similar_districts(seats)
    for k, seat in enumerate(seats):
        wc = Counter(winners[:, k])
        pc = Counter(pairs[:, k])
        row = {"district_id": seat.district_id, "district": seat.name, "region": seat.region,
               "winner_2022": seat.winner22, "anchor_formal_weight": float(seat.formal),
               "ind_candidate": seat.ind_name, "ind_mode": seat.ind_mode}
        for i, g in enumerate(GROUPS):
            v = prim[:, k, i]
            row[f"primary_{g}_mean"] = float(v.mean())
            row[f"primary_{g}_lo90"] = float(np.quantile(v, 0.05))
            row[f"primary_{g}_hi90"] = float(np.quantile(v, 0.95))
        for g in GROUPS:
            row[f"win_{g}"] = wc.get(g, 0) / D
        row["win_LNP"] = row["win_LIB"] + row["win_NAT"]
        row["likely_winner"] = max(["ALP", "LNP", "GRN", "ON", "IND", "OTH"], key=lambda g: row[f"win_{g}"])
        top_pairs = pc.most_common(3)
        row["final_pairs"] = [{"pair": p, "probability": c / D} for p, c in top_pairs]
        row["on_ratio_federal"] = float(seat.on_ratio)
        row["on_ratio_embedding"] = float(seat.on_ratio_emb)
        row["similar"] = sim.get(seat.district_id, [])
        row["retiring_party"] = seat.retiring_party
        row["on_ratio_mrp"] = seat.on_ratio_mrp
        row["on_pattern"] = seat.on_pattern
        row["sophomore_party"] = seat.sophomore_party
        row["fed_rel_swing"] = float(seat.fed_rel)
        row["stands"] = {k: bool(v) for k, v in seat.stands.items()}
        t = tpp[:, k]
        if np.isfinite(t).any():
            row["tpp_alp_mean"] = float(np.nanmean(t))
            row["tpp_alp_lo90"] = float(np.nanquantile(t, 0.05))
            row["tpp_alp_hi90"] = float(np.nanquantile(t, 0.95))
        else:
            row["tpp_alp_mean"] = row["tpp_alp_lo90"] = row["tpp_alp_hi90"] = None
        seat_rows.append(row)
    # seat counts per draw by coalition-grouped winner
    cats = ["ALP", "LNP", "GRN", "ON", "IND", "OTH"]
    counts = np.zeros((D, len(cats)), dtype=int)
    for j, c in enumerate(cats):
        if c == "LNP":
            counts[:, j] = ((winners == "LIB") | (winners == "NAT")).sum(axis=1)
        else:
            counts[:, j] = (winners == c).sum(axis=1)
    maj = cfg["majority"]
    seat_dist = {}
    for j, c in enumerate(cats):
        v = counts[:, j]
        seat_dist[c] = {"mean": float(v.mean()), "lo90": int(np.quantile(v, 0.05)), "hi90": int(np.quantile(v, 0.95)),
                        "median": int(np.median(v)), "p_majority": float((v >= maj).mean()),
                        "histogram": {int(x): int(n) for x, n in sorted(Counter(v.tolist()).items())}}
    seat_leads = seat_lead_probabilities(counts, cats)
    p_hung = float((counts.max(axis=1) < maj).mean())
    # statewide 2PP from expected counts where defined
    formal = np.array([s.formal for s in seats], dtype=float)
    state_primary = statewide_primary_draws(prim, formal)
    valid = np.isfinite(tpp)
    state_tpp = (np.nan_to_num(tpp) * formal[None, :] * valid).sum(axis=1) / (formal[None, :] * valid).sum(axis=1)
    pi = {p: i for i, p in enumerate(PARTIES)}
    edition = {
        "schema": "vicmodel.forecast-v1.edition",
        "model": f"Model v{cfg.get('model_version', '1')}",
        "reporting_revision": "district-aggregate-v1",
        "government_summary_revision": "explicit-seat-ties-v1",
        "count_revision": "reference-oriented-preference-noise-v1",
        "final_pair_share_definition": "share of first group in sorted pair, not necessarily the winner",
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "election_day": cfg["election_day"],
        "draws": D,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "poll_bias": {p: {"mean": float(poll_bias(agg, cfg)[0][i]), "sd": float(poll_bias(agg, cfg)[1][i])} for i, p in enumerate(PARTIES)},
        "swing_scales": {p: {"regional_sd": float(swing_scales(cfg)[0][i]), "seat_sd": float(swing_scales(cfg)[1][i])} for i, p in enumerate(PARTIES)},
        "polls": {"n": agg.n_polls, "last_fieldwork": agg.last_fieldwork.isoformat(), "days_ahead": agg.days_ahead,
                  "level": {p: float(agg.level[i]) for i, p in enumerate(PARTIES)},
                  "house_effects": {h: {p: float(v[i]) for i, p in enumerate(PARTIES)} for h, v in agg.house.items()}},
        "statewide": {
            "primary": {p: {"mean": float(state_primary[:, pi[p]].mean()), "lo90": float(np.quantile(state_primary[:, pi[p]], 0.05)),
                            "hi90": float(np.quantile(state_primary[:, pi[p]], 0.95))} for p in PARTIES},
            "poll_target_primary": {p: {"mean": float(state[:, pi[p]].mean()), "lo90": float(np.quantile(state[:, pi[p]], 0.05)),
                                        "hi90": float(np.quantile(state[:, pi[p]], 0.95))} for p in PARTIES},
            "primary_weight_basis": "fixed anchor election formal votes; Narracan uses its separate supplementary event; no target-election volume conditioning",
            "tpp_alp": {"mean": float(state_tpp.mean()), "lo90": float(np.quantile(state_tpp, 0.05)),
                        "hi90": float(np.quantile(state_tpp, 0.95))},
            "seats": seat_dist, **seat_leads, "p_hung": p_hung, "majority_threshold": maj,
        },
        "seats": seat_rows,
        "assumptions": [
            "All published 2026 statewide polls; nominal n divided by 1.5; house effects shrunk toward zero; a shared 2 pp election-wide error per party.",
            "Seat primaries: 2022 result plus proportional swing in the non-One-Nation remainder, 2 pp regional and 3 pp seat innovations.",
            "One Nation seat share follows the 2025 federal One Nation district pattern (70% weight) with 25% seat and 15% regional lognormal noise; One Nation assumed to contest every seat.",
            "Majors assumed to stand everywhere before nominations close; LIB/NAT presence from the ABC provisional roster.",
            "Preference flows from 2022 Victorian distributions, 2025 federal Victorian One Nation flows and declared rates for One Nation finalists, with Beta uncertainty.",
            f"{'Half' if float(cfg.get('federal_swing_coef') or 0) == 0.5 else format(float(cfg.get('federal_swing_coef') or 0), '.0%')} of each district's 2022-to-2025 federal relative two-party swing (AEC booths on current districts) is carried into its Labor-v-Coalition split; a retiring member's party loses {100 * cfg.get('retirement_effect', 0.0):g} points and a first-term member's party gains {100 * cfg.get('sophomore_effect', 0.0):g} points.",
            "No validation of these intervals against outcomes exists; this is a model forecast, not a calibrated one.",
        ],
    }
    if cfg.get("ind_base_top_only"):   # v1.20
        edition["assumptions"].insert(-1, (
            "Independents: one who stands again starts from their own 2022 vote and keeps a median "
            f"{np.exp(float(cfg.get('ind_retention_log_mean') or 0.0)):.0%} of it, with a wide spread; the leading new independent "
            f"in a seat takes a median of about {cfg['new_ind_share']:.0%}, occasionally far more (Victorian results 2006 to 2022)."))
    if any(t.get("col") == "oth_count" for t in cfg.get("seat_shifts") or []):
        edition["assumptions"].insert(-1, (
            "The number of independents and minor-party candidates on the ballot, against 2022, moves the seat's other vote "
            "by 1.5 to 2 points a candidate (capped at three), taken from the other candidates in proportion."))
    if cfg.get("coalition_recommends_on_p") is not None and float(cfg["coalition_recommends_on_p"]) >= 1.0:
        edition["assumptions"].insert(-1, (
            f"Coalition preferences: the Coalition puts One Nation above Labor (announced 10 November; the Nationals' registered "
            f"cards do so) and its voters split {1 - cfg['lnp_to_alp_vs_on_recommended']:.0%} to One Nation in Labor-v-One-Nation "
            "counts (South Australia 2026 under the same recommendation), except where a seat's card differs."))
    elif cfg.get("coalition_recommends_on_p") is not None:
        edition["assumptions"].insert(-1, (
            f"Coalition preferences: one shared decision per simulated election; with probability {cfg['coalition_recommends_on_p']:.2f} "
            f"the Coalition puts One Nation above Labor and its voters split {1 - cfg['lnp_to_alp_vs_on_recommended']:.0%} to One Nation in "
            f"Labor-v-One-Nation counts (South Australia 2026), otherwise {1 - cfg['lnp_to_alp_vs_on_open']:.0%} (Victorian poll pairings); "
            "Nationals seats always recommend. No 2026 cards were registered when this edition ran."))
    if cfg.get("lib_alp_over_grn_p") is not None:
        edition["assumptions"].insert(-1, liberal_order_assumption(cfg))
    for d, rate in sorted((cfg.get("seat_lnp_alp_vs_on_rate") or {}).items()):
        edition["assumptions"].insert(-1, f"{d.replace('-', ' ').title()}: Coalition preferences reach Labor over One Nation at "
                                          f"{rate:.0%} because the seat's Liberal card differs from the statewide decision.")
    if cfg.get("seat_flow_adjust"):
        seats_adj = ", ".join(s.replace("-", " ").title() for s in sorted(cfg["seat_flow_adjust"]))
        edition["assumptions"].insert(-1, f"Seat-specific card evidence shifts preference rates in {seats_adj} "
                                          "(Labor open tickets: Labor ballots follow their unguided rather than directed preferences).")
    if cfg.get("disendorsed"):
        dis = ", ".join(f"{d.replace('-', ' ').title()} ({g})" for d, g in sorted(cfg["disendorsed"].items()))
        edition["assumptions"].insert(-1, f"Disendorsed after the ballot draw, still on the ballot under the party name: {dis}; "
                                          f"that candidate's share is cut {cfg['disendorsed_share_cut']:.0%} (sd {cfg['disendorsed_share_cut_sd']:.0%}).")
    for obs in cfg.get("extra_seat_on_observations") or []:
        if cfg.get("use_morgan_seat_poll", True):
            edition["assumptions"].insert(-1, f"Seat poll: {obs.get('label', ', '.join(obs['districts']))}, One Nation "
                                              f"{obs['on_share']:.0%} with statewide polling about {obs['state_on_at_time']:.0%}, "
                                              f"a soft observation (weight {obs['weight']}) of the seat's One Nation ratio.")
    if cfg.get("on_poll_error_file"):
        edition["assumptions"].insert(-1, "One Nation's shared poll error comes from its final-poll record in Queensland 2017 and 2024, "
                                          "the 2025 federal election and South Australia 2026 (mean overstatement 0.7 points, half corrected; sd floor 2 points).")
    if cfg.get("trend_local_linear"):
        mom = float(cfg.get("trend_momentum") or 0.0)
        edition["assumptions"].insert(-1, f"Poll trend: a straight line through the polls weighted by a {cfg['trend_bandwidth_days']:g}-day "
                                          "Gaussian kernel, read at the latest fieldwork date"
                                          + (f" and carried {mom:.0%} of the way to polling day." if mom else
                                             " (a weighted average lags a moving trend at the edge of the data)."))
    shape = cfg.get("swing_scale_party") or {}
    if shape.get("ALP") == "additive" and shape.get("LNP") == "additive":   # v1.22
        floor = float(cfg.get("swing_additive_floor_frac") or 0.0)
        edition["assumptions"].insert(-1, "Swing shape: Labor's and the Coalition's statewide swings are the same number of points in every seat"
                                          + (f" (a party keeps at least {floor:.0%} of its previous vote in a seat)" if floor else "")
                                          + "; the Greens' and others' swings are half points, half in proportion to their seat vote.")
    if cfg.get("seat_sd_country_mult") or cfg.get("seat_sd_melbourne_mult"):   # v1.22
        cm, mm = float(cfg.get("seat_sd_country_mult") or 1.0), float(cfg.get("seat_sd_melbourne_mult") or 1.0)
        edition["assumptions"].insert(-1, f"A seat's own swing away from the statewide swing is {abs(cm - 1):.0%} "
                                          f"{'wider' if cm >= 1 else 'narrower'} in country seats and {abs(mm - 1):.0%} "
                                          f"{'wider' if mm >= 1 else 'narrower'} in Melbourne seats (the model's misses from 2006 to 2022 "
                                          "were wider in the country at every election).")
    if cfg.get("new_ind_margin_log") or cfg.get("new_ind_open_log"):   # v1.22
        edition["assumptions"].insert(-1, "A new independent's starting vote is higher the safer the seat and where the sitting member is "
                                          "retiring (230 new-independent contests, 2006 to 2022).")
    if any(t.get("col") == "retire_gap" for t in cfg.get("seat_shifts") or []):   # v1.22
        edition["assumptions"].insert(-1, "Where a member retires, the seat's Labor-v-Coalition split moves by about 0.18 of the gap between "
                                          "the retiring party's state and federal vote in the seat (measured on the 2006 to 2022 backtests; "
                                          "no clear mechanism, so given a wide error).")
    if any(t.get("col") == "a_grn_young" for t in cfg.get("seat_shifts") or []):   # v1.23
        edition["assumptions"].insert(-1, "The Greens' vote is moved toward seats with more people aged 20 to 34 and away from older seats "
                                          "(about half a point per standard deviation; statewide total unchanged), because the Greens beat the "
                                          "model in young seats in four of the five elections from 2006 to 2022.")
    if cfg.get("region_split_file"):
        edition["assumptions"].insert(-1, "Each party's Melbourne/country balance is pulled toward published poll crosstabs "
                                          "(Roy Morgan, DemosAU), weighted by sample size, age and the model's own uncertainty; statewide totals are unchanged.")
        if cfg.get("region_split_rural_only_parties"):
            names = {"ALP": "Labor", "LNP": "the Coalition", "GRN": "the Greens", "ON": "One Nation"}
            who = " and ".join(names.get(p, p) for p in cfg["region_split_rural_only_parties"])
            n_rural = sum(1 for s in seats if not is_melbourne(s) and s.district_id not in PROVINCIAL_CITY_DISTRICTS)
            edition["assumptions"].insert(-1, f"For {who} that country correction falls on the {n_rural} rural seats, and the Geelong, "
                                              "Ballarat, Bendigo and Macedon seats move with Melbourne: past state and federal results and the "
                                              "YouGov MRP put the model's country shortfall in the rural seats, not the provincial cities.")
    return edition, counts


def run_forecast(root: Path, as_of: date, cfg: dict | None = None, roster_path: Path | None = None,
                 polls_path: Path | None = None, anchor_path: Path | None = None, on_path: Path | None = None):
    cfg = {**DEFAULTS, **(cfg or {})}
    cfg["_root"] = root
    rng = np.random.default_rng(cfg["seed"])
    _EMPIRICAL_FLOWS["table"] = load_empirical_flows(root / cfg["flows_file"], cfg) if cfg.get("flows_file") else {}
    polls = pd.read_csv(polls_path or root / "data/curated/polls-2026-all.csv", dtype=str)
    agg = aggregate_polls(polls, as_of, cfg)
    state = draw_statewide(agg, cfg, rng)
    state = leader_state_adjust(state, cfg)   # leadership hypotheses (off by default): own stream, no other draw moves
    if roster_path is None and cfg.get("use_default_roster", True):
        roster_path = root / "data/curated/vic2026_la/abc-provisional-20261027/candidates.json"
    anchor_path = anchor_path or root / "data/curated/forecast-v1/anchor-2022.csv"
    seats = load_seats(root, roster_path, cfg, anchor_path, on_path)
    anchor = pd.read_csv(anchor_path)
    F = anchor.formal.sum()
    state22 = {g: float((anchor[f"share_{g}"] * anchor.formal).sum() / F) for g in GROUPS}
    prim = seat_primaries(seats, state, state22, cfg, rng)
    # v1.17 seat factors: their own stream, before the Melbourne/country calibration so regional totals still calibrate
    prim, factor_report = apply_seat_factors(prim, seats, cfg, root, np.random.default_rng([int(cfg["seed"]), 117]))
    pi_ = {p: i for i, p in enumerate(PARTIES)}   # statewide Labor-v-Coalition swing per draw, from first preferences
    state_swing = 0.5 * ((state[:, pi_["ALP"]] - state22["ALP"]) - (state[:, pi_["LNP"]] - state22["LIB"] - state22["NAT"]))
    prim, shift_report = apply_seat_shifts(prim, seats, cfg, root, np.random.default_rng([int(cfg["seed"]), 120]), state_swing)
    if cfg.get("gender_gap_change"):   # gender hypotheses (off by default): own stream, no other draw moves
        prim = gender_apply_gap(prim, seats, cfg, root, np.random.default_rng([int(cfg["seed"]), 1119]))
    if cfg.get("grn_bt_alp_fall_transfer"):   # Greens backtests (21 Nov, off by default): own stream, no other draw moves
        prim = grn_bt_alp_fall(prim, seats, state, state22, cfg, np.random.default_rng([int(cfg["seed"]), 1121]))
    # v1.12 components draw from their own stream so switching one off leaves every other draw unchanged
    rng_v112 = np.random.default_rng([int(cfg["seed"]), 112])
    prim, split_report = region_split_calibration(prim, seats, cfg, as_of, rng_v112)
    recommend = None
    if cfg.get("coalition_recommends_on_p") is not None:
        recommend = rng_v112.random(prim.shape[0]) < float(cfg["coalition_recommends_on_p"])
    # v1.13 components: their own stream again, so each can be switched off without moving any earlier draw
    rng_v113 = np.random.default_rng([int(cfg["seed"]), 113])
    prim, disendorsed_report = disendorse(prim, seats, cfg, rng_v113)
    if cfg.get("seat_primary_moves"):   # Tally Room idea 3 (off by default): own stream, no other draw moves
        prim = apply_seat_primary_moves(prim, seats, cfg, np.random.default_rng([int(cfg["seed"]), 119]))
    lib_first = None
    if cfg.get("lib_alp_over_grn_p") is not None:
        lib_first = rng_v113.random(prim.shape[0]) < float(cfg["lib_alp_over_grn_p"])
    if cfg.get("primaries_only"):   # primaries only: no preference count
        return None, {"state": state, "prim": prim, "seats": seats, "agg": agg, "recommend": recommend, "lib_first": lib_first}
    winners, pairs, pair_share, tpp = simulate(seats, prim, cfg, rng, recommend, lib_first)
    edition, counts = summarise(seats, state, prim, winners, pairs, pair_share, tpp, agg, cfg, as_of)
    if split_report:
        edition["statewide"]["region_split"] = split_report
    if recommend is not None:
        edition["statewide"]["preference_scenarios"] = preference_scenario_summary(counts, recommend, winners, seats, cfg)
    if lib_first is not None:
        edition["statewide"]["liberal_labor_greens_order"] = scenario_summary(
            counts, lib_first, winners, seats, cfg, ("labor_above_greens", "greens_above_labor"),
            {"p_labor_above_greens": float(cfg["lib_alp_over_grn_p"]),
             "lnp_to_alp_vs_grn": {"labor_above_greens": cfg["lnp_to_alp_vs_grn_directed"],
                                   "greens_above_labor": cfg["lnp_to_alp_vs_grn_reversed"],
                                   "open_ticket": cfg["lnp_to_alp_vs_grn_open"]},
             "open_ticket_seats": sorted(cfg.get("lib_open_alp_grn_seats") or []),
             **({"registered_cards": cfg["lib_cards_registered"]} if cfg.get("lib_cards_registered") else {})},
            parties=("ALP", "LNP", "GRN"))
    if disendorsed_report:
        edition["statewide"]["disendorsed"] = disendorsed_report
    if shift_report:
        edition["statewide"]["seat_shifts"] = shift_report
    if factor_report:
        edition["statewide"]["seat_factors"] = factor_report
        idx = factor_report["_index"]
        for row in edition["seats"]:
            for name, vals in idx.items():
                row[name] = vals.get(row["district_id"])
        del factor_report["_index"]
    return edition, {"state": state, "prim": prim, "winners": winners, "pairs": pairs, "tpp": tpp, "counts": counts,
                     "seats": seats, "agg": agg, "recommend": recommend, "lib_first": lib_first}


def apply_seat_factors(prim: np.ndarray, seats: list, cfg: dict, root: Path, rng: np.random.Generator):
    """v1.17: shared demographic swing factors. For each factor f with seat index z_s (standardised Census composite),
    one coefficient b_f ~ Normal(mean_f, sd_f) is drawn per simulated election and b_f * z_s of the seat's vote moves
    from the Coalition to Labor (negative b: the other way). Seats move together by demographic type, in a direction
    that differs from election to election (2018 and 2022 backtests). Applied where both Labor and the Coalition stand,
    and only in classic Labor-v-Coalition anchor seats when seat_factor_classic_only. Returns (prim, report)."""
    factors = cfg.get("seat_factors") or {}
    path = root / cfg["seat_index_file"] if cfg.get("seat_index_file") else None
    if not factors or not path or not path.exists():
        return prim, {}
    tab = pd.read_csv(path).set_index("district_id")
    gi = {g: i for i, g in enumerate(GROUPS)}
    D, S = prim.shape[0], prim.shape[1]
    prim = prim.copy()
    shift = np.zeros((D, S))
    rep = {"index_file": cfg["seat_index_file"], "factors": {}, "_index": {}}
    for name, spec in factors.items():
        z = np.array([float(tab[name].get(s.district_id, 0.0)) for s in seats])
        b = rng.normal(float(spec["mean"]), float(spec["sd"]), D)
        shift += b[:, None] * z[None, :]
        rep["factors"][name] = {"mean": float(spec["mean"]), "sd": float(spec["sd"]),
                                "note": spec.get("note", "")}
        rep["_index"][name] = {s.district_id: float(z[k]) for k, s in enumerate(seats)}
    classic_only = cfg.get("seat_factor_classic_only", False)
    moved = 0
    for k, s in enumerate(seats):
        alp = prim[:, k, gi["ALP"]]
        lnp = prim[:, k, gi["LIB"]] + prim[:, k, gi["NAT"]]
        if not (s.stands.get("ALP") and (s.stands.get("LIB") or s.stands.get("NAT"))):
            continue
        if classic_only and not getattr(s, "classic", True):
            continue
        x = shift[:, k]
        x = np.clip(x, -(alp - 0.002), lnp - 0.002)   # keep both shares positive
        frac_lib = np.where(lnp > 0, prim[:, k, gi["LIB"]] / np.where(lnp > 0, lnp, 1), 0.0)
        prim[:, k, gi["ALP"]] = alp + x
        prim[:, k, gi["LIB"]] -= x * frac_lib
        prim[:, k, gi["NAT"]] -= x * (1 - frac_lib)
        moved += 1
    rep["seats_moved"] = moved
    return prim, rep


def apply_seat_shifts(prim: np.ndarray, seats: list, cfg: dict, root: Path, rng: np.random.Generator,
                      state_swing: np.ndarray | None = None):
    """v1.20: seat-level shifts from information that is public before polling day (the federal vote in the district,
    the number of candidates). Each term reads one column of a district file; `center` removes its formal-weighted mean
    so statewide totals stay with the polls. A term with `times_state_swing` is multiplied by each draw's statewide
    Labor-v-Coalition swing (seats that move more or less than the state). Returns (prim, report)."""
    terms = cfg.get("seat_shifts") or []
    if not terms:
        return prim, None
    gi = {g: i for i, g in enumerate(GROUPS)}
    D = prim.shape[0]
    prim = prim.copy()
    formal = np.array([s.formal for s in seats], float)
    report = []
    for term in terms:
        tab = pd.read_csv(root / term["file"]).set_index("district_id")[term["col"]].astype(float)
        x = np.array([tab.get(s.district_id, np.nan) for s in seats])
        have = np.isfinite(x)
        if term.get("classic_only"):
            have &= np.array([bool(getattr(s, "classic", True)) for s in seats])
        if term.get("center", True) and have.any():
            x = x - (x[have] * formal[have]).sum() / formal[have].sum()
        x = np.where(have, x, 0.0)
        b = rng.normal(float(term["coef"]), float(term.get("coef_sd", 0.0)), D)
        if term.get("times_state_swing") and state_swing is not None:
            b = b * state_swing
        target = term["target"]
        moved = 0
        for k, s in enumerate(seats):
            if x[k] == 0.0:
                continue
            sh = b * x[k]
            if target == "ALP_LNP":
                if not (s.stands.get("ALP") and (s.stands.get("LIB") or s.stands.get("NAT"))):
                    continue
                alp = prim[:, k, gi["ALP"]]
                lnp = prim[:, k, gi["LIB"]] + prim[:, k, gi["NAT"]]
                sh = np.clip(sh, -(alp - 0.002), lnp - 0.002)
                frac_lib = np.where(lnp > 0, prim[:, k, gi["LIB"]] / np.where(lnp > 0, lnp, 1), 0.0)
                prim[:, k, gi["ALP"]] = alp + sh
                prim[:, k, gi["LIB"]] -= sh * frac_lib
                prim[:, k, gi["NAT"]] -= sh * (1 - frac_lib)
            else:
                cols = [gi["LIB"], gi["NAT"]] if target == "LNP" else [gi[target]]
                cur = prim[:, k, cols].sum(axis=1)
                if not (cur > 0).any():
                    continue   # the group does not stand here
                rest = prim[:, k, :].sum(axis=1) - cur
                new = np.clip(cur + sh, 0.002, np.maximum(cur + rest - 0.01, 0.002))
                scale_rest = (rest - (new - cur)) / np.maximum(rest, 1e-9)
                scale_cur = new / np.maximum(cur, 1e-9)
                mask = np.ones(len(GROUPS), bool)
                mask[cols] = False
                prim[:, k, mask] = prim[:, k, mask] * scale_rest[:, None]
                prim[:, k, ~mask] = prim[:, k, ~mask] * scale_cur[:, None]
            moved += 1
        report.append({**{k: term[k] for k in ("file", "col", "target", "coef")}, "coef_sd": float(term.get("coef_sd", 0.0)),
                       "seats_moved": moved, "value_sd": float(np.std(x[have])) if have.any() else 0.0})
    return prim, report


def gender_apply_gap(prim: np.ndarray, seats: list, cfg: dict, root: Path, rng: np.random.Generator) -> np.ndarray:
    """Gender hypotheses 2 and 4 (off by default): a party whose gender gap (women minus men, points) has changed since the
    anchor election gains gap x (seat women's share - formal-weighted mean) in each seat; one gap drawn per simulated
    election from N(mean, sd). The gain is taken from the other groups in proportion, so statewide totals barely move.
    Seats missing from the file are not moved."""
    tab = pd.read_csv(root / cfg["gender_gap_file"]).set_index("district_id")[cfg.get("gender_gap_col") or "f_adult"].astype(float)
    w = np.array([tab.get(s.district_id, np.nan) for s in seats])
    have = np.isfinite(w)
    if not have.any():
        return prim
    formal = np.array([s.formal for s in seats], float)
    x = np.where(have, w - (w[have] * formal[have]).sum() / formal[have].sum(), 0.0)
    gi = {g: i for i, g in enumerate(GROUPS)}
    prim = prim.copy()
    D = prim.shape[0]
    for party, spec in cfg["gender_gap_change"].items():
        mean, sd = (spec, 0.0) if np.isscalar(spec) else (spec[0], spec[1] if len(spec) > 1 else 0.0)
        gap = rng.normal(float(mean), float(sd), D) / 100.0
        cols = [gi["LIB"], gi["NAT"]] if party == "LNP" else [gi[party]]
        mask = np.ones(len(GROUPS), bool)
        mask[cols] = False
        for k in np.flatnonzero(x != 0.0):
            cur = prim[:, k, cols].sum(axis=1)
            if not (cur > 0).any():
                continue
            rest = prim[:, k, :].sum(axis=1) - cur
            new = np.where(cur > 0, np.clip(cur + gap * x[k], 0.002, np.maximum(cur + rest - 0.01, 0.002)), cur)
            prim[:, k, mask] *= ((rest - (new - cur)) / np.maximum(rest, 1e-9))[:, None]
            prim[:, k, ~mask] *= (new / np.maximum(cur, 1e-9))[:, None]
    return prim


def apply_seat_primary_moves(prim: np.ndarray, seats: list, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Tally Room idea 3 (off by default): in named seats move a drawn share of the first preferences from one group to
    another ({district: [[from, to, mean, sd], ...]}), e.g. a new party whose name resembles the Greens' cause taking
    Greens votes. The amount is drawn per simulated election, clipped to [0, the from-group's share - 0.002]."""
    prim = prim.copy()
    gi = {g: i for i, g in enumerate(GROUPS)}
    idx = {s.district_id: k for k, s in enumerate(seats)}
    for d, moves in (cfg.get("seat_primary_moves") or {}).items():
        if d not in idx:
            continue
        k = idx[d]
        for src, dst, mean, sd in moves:
            have = prim[:, k, gi[src]]
            if not (have > 0).any():
                continue
            x = np.clip(rng.normal(float(mean), float(sd), prim.shape[0]), 0.0, np.maximum(have - 0.002, 0.0))
            prim[:, k, gi[src]] = have - x
            prim[:, k, gi[dst]] = prim[:, k, gi[dst]] + x
    return prim


def disendorse(prim: np.ndarray, seats: list, cfg: dict, rng: np.random.Generator):
    """v1.13: a candidate disendorsed after the ballot draw keeps the party name on the ballot but loses the campaign.
    Cut that group's seat share by a drawn relative amount and give it to the other candidates in proportion."""
    rules = cfg.get("disendorsed") or {}
    if not rules:
        return prim, None
    prim = prim.copy()
    gi = {g: i for i, g in enumerate(GROUPS)}
    report = {}
    for k, s in enumerate(seats):
        g = rules.get(s.district_id)
        if not g:
            continue
        cut = np.clip(rng.normal(cfg["disendorsed_share_cut"], cfg["disendorsed_share_cut_sd"], prim.shape[0]), 0.0, 0.5)
        x = prim[:, k, :]
        lost = x[:, gi[g]] * cut
        rest = x.sum(axis=1) - x[:, gi[g]]
        mask = np.arange(len(GROUPS)) != gi[g]
        x[:, mask] = x[:, mask] * (1 + (lost / np.maximum(rest, 1e-9)))[:, None]
        x[:, gi[g]] = x[:, gi[g]] - lost
        report[s.district_id] = {"group": g, "mean_share_lost": float(lost.mean())}
    return prim, report


def preference_scenario_summary(counts: np.ndarray, recommend: np.ndarray, winners: np.ndarray, seats: list,
                                cfg: dict) -> dict:
    """Seat totals and government outcomes conditional on the drawn Coalition preference decision (v1.12)."""
    return scenario_summary(counts, recommend, winners, seats, cfg, ("recommended", "open"),
                            {"p_coalition_recommends_on": float(cfg["coalition_recommends_on_p"]),
                             "lnp_to_alp_vs_on": {"recommended": cfg["lnp_to_alp_vs_on_recommended"],
                                                  "open": cfg["lnp_to_alp_vs_on_open"]}})


def scenario_summary(counts: np.ndarray, flag: np.ndarray, winners: np.ndarray, seats: list, cfg: dict,
                     names: tuple, header: dict, parties=("ALP", "LNP", "ON")) -> dict:
    """Seat totals, government outcomes and seat-level shifts conditional on a shared per-election draw."""
    cats = ["ALP", "LNP", "GRN", "ON", "IND", "OTH"]
    maj = cfg.get("majority", 45)
    coal = np.where(np.isin(winners, ["LIB", "NAT"]), "LNP", winners)
    out = {**header, "scenarios": {}}
    for name, mask in ((names[0], flag), (names[1], ~flag)):
        if not mask.any():
            continue
        c = counts[mask]
        flips = {}
        other = ~mask
        if other.any():
            for k, s in enumerate(seats):
                for p in parties:
                    d = float((coal[mask, k] == p).mean() - (coal[other, k] == p).mean())
                    if abs(d) >= 0.05:
                        flips.setdefault(s.district_id, {})[p] = round(d, 3)
        out["scenarios"][name] = {
            "draws": int(mask.sum()),
            "seats_mean": {p: float(c[:, i].mean()) for i, p in enumerate(cats)},
            "p_majority": {p: float((c[:, i] >= maj).mean()) for i, p in enumerate(cats) if p in ("ALP", "LNP")},
            "p_hung": float(((c < maj).all(axis=1)).mean()),
            "seat_probability_shift_vs_other_scenario": flips,
        }
    return out


# ---- Greens backtest options (21 November 2026, docs/greens-backtests-20261121.md). All off by default: --------------
# with none of these keys set every function below returns its input unchanged and uses no random numbers.
#   grn_bt_poll_bias {"overstatement_pp": x, "sd_pp": s}   the Greens' shared poll error: mean correction -x points (polls
#       overstate the Greens by x) and sd s points, replacing the Victorian-history values for the Greens only (H3).
#   grn_bt_flow_means {"final": {"ALP-GRN": {"OTH": r}}, "weights": {"OTH": {"ALP": a, "GRN": g}, ...}}   mean preference
#       rates: final-pair rates (to the first group named) and transfer weights, replacing the declared table's means
#       (H2: minor-party and independent ballots favour the Greens over Labor more than the declared table assumes).
#       Only existing keys are replaced, so the number of random draws, and every other draw, is unchanged.
#   grn_bt_alp_fall_transfer {"grn": a, "alp": b, "sd_frac": f}   in each simulated election, with Labor's statewide
#       first preference down F on the anchor, the Greens gain a * F * (seat's anchor Greens share - formal-weighted mean)
#       and Labor gains b * F * (the same); the other groups give or take the rest in proportion (H1: in Labor-collapse
#       elections Labor holds up, and the Greens gain less, where the Greens were already strong). Coefficients are
#       drawn once per simulated election as c * (1 + f z) on their own random stream.
def grn_bt_poll_bias(mean: np.ndarray, sd: np.ndarray, cfg: dict):
    opt = cfg.get("grn_bt_poll_bias")
    if not opt:
        return mean, sd
    mean, sd = mean.copy(), sd.copy()
    i = PARTIES.index("GRN")
    if opt.get("overstatement_pp") is not None:
        mean[i] = -float(opt["overstatement_pp"]) / 100.0
    if opt.get("sd_pp") is not None:
        sd[i] = float(opt["sd_pp"]) / 100.0
    return mean, sd


def grn_bt_flow_tables(cfg: dict, final_rates: dict, transfer: dict):
    opt = cfg.get("grn_bt_flow_means")
    if not opt:
        return final_rates, transfer
    final_rates = {k: dict(v) for k, v in final_rates.items()}
    transfer = {o: dict(w) for o, w in transfer.items()}
    for key, origins in (opt.get("final") or {}).items():
        a, b = key.split("-")
        for o, m in origins.items():
            if (a, b) in final_rates and o in final_rates[(a, b)]:
                final_rates[(a, b)][o] = float(m)
            elif (b, a) in final_rates and o in final_rates[(b, a)]:
                final_rates[(b, a)][o] = 1.0 - float(m)
            else:
                raise KeyError(f"grn_bt_flow_means: no declared final-pair rate {key} for {o}")
    for o, dests in (opt.get("weights") or {}).items():
        for d, w in dests.items():
            if d not in transfer.get(o, {}):
                raise KeyError(f"grn_bt_flow_means: no declared transfer weight {o}->{d}")
            transfer[o][d] = float(w)
    return final_rates, transfer


def grn_bt_alp_fall(prim: np.ndarray, seats: list, state: np.ndarray, state22: dict, cfg: dict,
                    rng: np.random.Generator) -> np.ndarray:
    opt = cfg["grn_bt_alp_fall_transfer"]
    gi = {g: i for i, g in enumerate(GROUPS)}
    D = prim.shape[0]
    fall = np.clip(state22["ALP"] - state[:, PARTIES.index("ALP")], 0.0, None)
    z = rng.normal(0, 1, D) * float(opt.get("sd_frac", 0.0))
    bg = float(opt.get("grn", 0.0)) * (1 + z) * fall
    ba = float(opt.get("alp", 0.0)) * (1 + z) * fall
    formal = np.array([s.formal for s in seats], float)
    g22 = np.array([s.share22.get("GRN", 0.0) for s in seats], float)
    use = np.array([bool(s.stands.get("GRN", True)) and bool(s.stands.get("ALP", True)) for s in seats])
    if not use.any():
        return prim
    x = np.where(use, g22 - (g22[use] * formal[use]).sum() / formal[use].sum(), 0.0)
    prim = prim.copy()
    others = [i for g, i in gi.items() if g not in ("ALP", "GRN")]
    for k in np.nonzero(use)[0]:
        tot = prim[:, k, :].sum(axis=1)
        g, a = prim[:, k, gi["GRN"]], prim[:, k, gi["ALP"]]
        rest = tot - g - a
        new_g = np.clip(g + bg * x[k], 0.002, None)
        new_a = np.clip(a + ba * x[k], 0.002, None)
        over = np.clip(new_g + new_a - (tot - 0.01), 0.0, None)   # keep at least one point for the other groups
        share_g = np.where(new_g + new_a > 0, new_g / (new_g + new_a), 0.5)
        new_g, new_a = new_g - over * share_g, new_a - over * (1 - share_g)
        scale = (tot - new_g - new_a) / np.maximum(rest, 1e-9)
        prim[:, k, others] = prim[:, k, others] * scale[:, None]
        prim[:, k, gi["GRN"]], prim[:, k, gi["ALP"]] = new_g, new_a
    return prim
# ---- end of Greens backtest options ------------------------------------------------------------------------------------


def similar_districts(seats: list, k: int = 5) -> dict:
    """For each seat, the k nearest districts in standardised embedding space (used for display and GIS export)."""
    if any(s.pc is None for s in seats):
        return {}
    P = np.stack([s.pc for s in seats])
    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1)
    out = {}
    for i, s in enumerate(seats):
        order = np.argsort(D2[i])
        out[s.district_id] = [{"district_id": seats[j].district_id, "district": seats[j].name, "distance": float(np.sqrt(D2[i, j]))}
                              for j in order if j != i][:k]
    return out


def write_edition(edition: dict, arrays: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "edition.json").write_text(json.dumps(edition, indent=1), encoding="utf-8")
    rows = []
    for r in edition["seats"]:
        flat = {k: v for k, v in r.items() if k not in ("final_pairs", "similar")}
        flat["similar"] = "; ".join(x["district"] for x in r.get("similar", []))
        flat["top_pair"] = r["final_pairs"][0]["pair"] if r["final_pairs"] else ""
        flat["top_pair_p"] = r["final_pairs"][0]["probability"] if r["final_pairs"] else ""
        rows.append(flat)
    pd.DataFrame(rows).to_csv(out_dir / "seats.csv", index=False)
    np.savez_compressed(out_dir / "draws.npz", state=arrays["state"], prim=arrays["prim"],
                        winners=arrays["winners"].astype(str), tpp=arrays["tpp"], counts=arrays["counts"],
                        district_ids=np.array([s.district_id for s in arrays["seats"]]),
                        count_categories=np.array(["ALP", "LNP", "GRN", "ON", "IND", "OTH"]),
                        anchor_formal_weights=np.array([s.formal for s in arrays["seats"]], dtype=float),
                        **({"coalition_recommends_on": arrays["recommend"]} if arrays.get("recommend") is not None else {}),
                        **({"liberal_labor_above_greens": arrays["lib_first"]} if arrays.get("lib_first") is not None else {}))
    arrays["agg"].detail.to_csv(out_dir / "polls-used.csv", index=False)
