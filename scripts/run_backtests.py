"""Backtest Model v1.25 on the Victorian state elections 2006-2022 and print the scores.

Each election is forecast from the previous election's results, the polls published before it and the information
the model would have had (candidates, sitting members), then scored against the declared winners. Settings are frozen
in configs/backtests-v1.25.json.

Usage: python scripts/run_backtests.py [--seeds 1] [--draws 2000] [--years 2006,2010,2014,2018,2022]
Writes runs/forecast-v1-backtest-<year>-v125-s<k>/ (edition, seats, scorecard) and prints winner Brier, log loss and
winners right per election and on average.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATS = ["ALP", "LNP", "GRN", "ON", "IND", "OTH"]   # party groups scored; Liberal and National count as one (LNP)


def score(year: str, out: Path) -> dict:
    """Winner Brier (sum over party groups, mean over seats), log loss (winner's probability floored at 0.005) and the
    number of seats whose most likely winner won, against the declared winners."""
    seats = pd.read_csv(out / "seats.csv").set_index("district_id")
    actual = pd.read_csv(ROOT / f"data/curated/forecast-v1/anchor-{year}.csv").set_index("district_id")
    common = actual.index.intersection(seats.index)
    P = seats.loc[common, [f"win_{c}" for c in CATS]].to_numpy(float)
    y = actual.loc[common, "winner_2022"].replace({"LIB": "LNP", "NAT": "LNP"}).map(CATS.index).to_numpy()
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1
    card = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    return {"brier": float(np.mean(((P - Y) ** 2).sum(1))), "log_loss": float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 0.005, 1)))),
            "correct": card["modal_winner_correct"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1, help="how many of the six frozen seeds to run")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--years", default="2006,2010,2014,2018,2022")
    a = ap.parse_args()
    spec = json.loads((ROOT / "configs/backtests-v1.25.json").read_text(encoding="utf-8"))
    rows = {}
    for y in a.years.split(","):
        e = spec["elections"][y]
        for k, seed in enumerate(spec["seeds"][: a.seeds]):
            tag = f"v125-s{k}"
            cmd = [sys.executable, str(ROOT / "scripts/backtest_v1.py"), "--anchor", str(e["anchor"]), "--target", str(e["target"]),
                   "--polls", e["polls"], "--as-of", e["as_of"], "--election-day", e["election_day"], "--draws", str(a.draws),
                   "--tag", tag, *[x for s in e["set"] + [f"seed={seed}"] for x in ("--set", s)]]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
            rows.setdefault(y, []).append(score(y, ROOT / f"runs/forecast-v1-backtest-{y}-{tag}"))
    keys = ("brier", "log_loss", "correct")
    print(f"{'election':10s} {'Brier':>8s} {'log loss':>9s} {'winners right':>14s}")
    means = {m: [] for m in keys}
    for y, cards in rows.items():
        v = {m: float(np.mean([c[m] for c in cards])) for m in keys}
        for m in keys:
            means[m].append(v[m])
        print(f"{y:10s} {v['brier']:8.4f} {v['log_loss']:9.4f} {v['correct']:14.1f}")
    print(f"{'mean':10s} {np.mean(means['brier']):8.4f} {np.mean(means['log_loss']):9.4f} {np.mean(means['correct']):14.1f}")


if __name__ == "__main__":
    main()
