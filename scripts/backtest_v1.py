"""Generic Model v1 backtest: anchor election A, polls to the eve of election B, scored against B.

Usage:
  python scripts/backtest_v1.py --anchor 2014 --target 2018 --polls data/curated/forecast-v1/polls-2018-all.csv --as-of 2018-11-23 --election-day 2018-11-24
  python scripts/backtest_v1.py --anchor 2018 --target 2022 --polls data/curated/forecast-v1/polls-2022-all.csv --as-of 2022-11-25 --election-day 2022-11-26
The 2018->2022 pair crosses a redistribution: only name-matched districts are scored and the anchor is the
old-boundary result, so expect extra noise. No One Nation, no 2026 rosters; 2014/2018 independents >= 8% assumed
to recontest. Writes runs/forecast-v1-backtest-<target>[-tag]/ with edition, seats and scorecard.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicmodel import forecast_v1 as fv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--polls", required=True)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--election-day", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", action="append", default=[], help="cfg override key=value (json value)")
    a = ap.parse_args()
    cfg = {"draws": a.draws, "election_day": a.election_day, "on_contests_all": False, "on_floor": 0.0,
           "use_morgan_seat_poll": False, "use_default_roster": False, "on_pattern_file": False,
           "assume_ind_recontest": True, "poll_error_exclude": str(a.target),
           "absent_party_base_file": None, "departed_ind_to_parties": 0.0,   # v1.8 roster features need known rosters; off in backtests
           # v1.9 ballot position: the target election's draw was public before polling day; the gain is the estimate
           # available at the time (2006-2014 pairs for 2018, 2006-2018 pairs for 2022)
           "ballot_base_file": f"data/curated/forecast-v1/ballot-top-{a.anchor}.csv",
           "ballot_now_file": f"data/curated/forecast-v1/ballot-top-{a.target}.csv",
           "ballot_top_gain": {2018: 0.019, 2022: 0.021}.get(a.target, 0.024),
           # v1.10 federal relative swing: the 2022 target can use the May 2022 federal result (2019->2022); no mapped
           # federal booths exist for the 2013->2016 term, so the 2018 backtest runs without it
           "federal_swing_col": {2022: "rel_swing_2019_2022_classic"}.get(a.target),
           "federal_swing_file": "data/curated/forecast-v1/federal-tpp-relative-swing.csv" if a.target == 2022 else None,
           # v1.19: seat swing spread pooled over the election pairs that ended by the anchor election (2006 has none
           # before its own pair and keeps the 2014->2018 measurement)
           "swing_dispersion_file": {2010: "data/curated/forecast-v1/swing-dispersion-pooled-2006.json",
                                     2014: "data/curated/forecast-v1/swing-dispersion-pooled-2010.json",
                                     2018: "data/curated/forecast-v1/swing-dispersion-pooled-2014.json",
                                     2022: "data/curated/forecast-v1/swing-dispersion-pooled-2018.json"}.get(
               a.target, "data/curated/forecast-v1/swing-dispersion.json")}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    edition, arrays = fv.run_forecast(ROOT, date.fromisoformat(a.as_of), cfg, polls_path=ROOT / a.polls,
                                      anchor_path=ROOT / f"data/curated/forecast-v1/anchor-{a.anchor}.csv")
    out = ROOT / f"runs/forecast-v1-backtest-{a.target}{('-' + a.tag) if a.tag else ''}"
    fv.write_edition(edition, arrays, out)
    actual = pd.read_csv(ROOT / f"data/curated/forecast-v1/anchor-{a.target}.csv").set_index("district_id")
    seats = pd.DataFrame([{k: v for k, v in r.items() if k not in ("final_pairs", "similar", "stands")} for r in edition["seats"]]).set_index("district_id")
    common = actual.index.intersection(seats.index)
    actual, seats = actual.loc[common], seats.loc[common]
    coal = {"LIB": "LNP", "NAT": "LNP"}
    act_w = actual["winner_2022"].replace(coal)
    pred_w = seats["likely_winner"]
    correct = (act_w == pred_w)
    cats = ["ALP", "LNP", "GRN", "ON", "IND", "OTH"]
    brier = float(np.mean([sum((float(seats.loc[d, f"win_{c}"]) - (1.0 if act_w[d] == c else 0.0)) ** 2 for c in cats) for d in common]))

    def grp_share(df, prefix):
        return pd.DataFrame({"ALP": df[f"{prefix}ALP"], "LNP": df[f"{prefix}LIB"] + df[f"{prefix}NAT"],
                             "GRN": df[f"{prefix}GRN"], "OTH": df[f"{prefix}OTH"] + df[f"{prefix}IND"] + df[f"{prefix}ON"]})
    pred_s = grp_share(seats.rename(columns={f"primary_{g}_mean": f"p_{g}" for g in fv.GROUPS}), "p_")
    act_s = grp_share(actual.rename(columns={f"share_{g}": f"a_{g}" for g in fv.GROUPS}), "a_")
    mae = (pred_s - act_s).abs().mean()
    sw = edition["statewide"]
    actual_counts = act_w.value_counts().to_dict()
    tpp_act = {}
    for d in common:
        if actual.loc[d, "final_pair_2022"] in ("ALP-LIB", "ALP-NAT"):
            g1, s1 = actual.loc[d, "final_g1"], actual.loc[d, "final_share_1"]
            tpp_act[d] = s1 if g1 == "ALP" else 1 - s1
    tpp_act = pd.Series(tpp_act)
    tpp_pred = seats["tpp_alp_mean"].reindex(tpp_act.index).astype(float)
    lo = seats["tpp_alp_lo90"].reindex(tpp_act.index).astype(float)
    hi = seats["tpp_alp_hi90"].reindex(tpp_act.index).astype(float)
    F = actual["formal"]
    card = {
        "anchor": a.anchor, "target": a.target, "as_of": a.as_of, "polls_used": edition["polls"]["n"], "draws": a.draws, "scored_seats": len(common),
        "poll_level": edition["polls"]["level"],
        "statewide_actual_primary": {c: float((act_s[c] * F).sum() / F.sum()) for c in act_s},
        "statewide_pred_primary": {p: sw["primary"][p]["mean"] for p in fv.PARTIES},
        "statewide_tpp_pred": sw["tpp_alp"],
        "modal_winner_correct": int(correct.sum()), "winner_brier": brier,
        "seat_counts_pred_mean": {c: sw["seats"][c]["mean"] for c in cats},
        "seat_counts_pred_90": {c: [sw["seats"][c]["lo90"], sw["seats"][c]["hi90"]] for c in cats},
        "seat_counts_actual_scored": {c: int(actual_counts.get(c, 0)) for c in cats},
        "primary_mae_pp": {c: 100 * float(v) for c, v in mae.items()}, "primary_mae_all_pp": 100 * float(mae.mean()),
        "classic_tpp_seats": len(tpp_act), "tpp_mae_pp": 100 * float((tpp_pred - tpp_act).abs().mean()),
        "tpp_bias_pp": 100 * float((tpp_pred - tpp_act).mean()),
        "tpp_90_coverage": float(((tpp_act >= lo) & (tpp_act <= hi)).mean()),
        "p_alp_majority": sw["seats"]["ALP"]["p_majority"],
        "missed_seats": [f"{d}: predicted {pred_w[d]} ({float(seats.loc[d, 'win_' + pred_w[d]]):.2f}), actual {act_w[d]}" for d in common if not correct[d]],
        "overrides": a.set,
    }
    (out / "scorecard.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in card.items() if k not in ("missed_seats", "poll_level")}, indent=1))
    print("missed:", *card["missed_seats"], sep="\n  ")


if __name__ == "__main__":
    main()
