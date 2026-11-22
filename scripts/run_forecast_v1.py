"""Run Model v1 and write a dated edition.

Usage:
  .venv\\Scripts\\python.exe -X utf8 scripts/run_forecast_v1.py [--as-of 2026-11-04] [--draws 2000] [--out runs/forecast-v1-<as_of>]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vicmodel.forecast_v1 import PARTIES, run_forecast, write_edition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--polls", type=Path, default=None, help="Explicit poll CSV for this edition")
    ap.add_argument("--config", type=Path, default=None, help="Declared model/data overrides for this fresh edition")
    a = ap.parse_args()
    as_of = date.fromisoformat(a.as_of)
    cfg = json.loads(a.config.read_text(encoding="utf-8")) if a.config else {}
    cfg["draws"] = a.draws
    if a.seed is not None:
        cfg["seed"] = a.seed
    out = Path(a.out) if a.out else ROOT / f"runs/forecast-v1-{as_of.isoformat()}"
    if (out / "edition.json").exists():
        ap.error(f"edition already exists at {out}; choose a fresh --out to preserve its predictions")
    edition, arrays = run_forecast(ROOT, as_of, cfg, polls_path=a.polls)
    write_edition(edition, arrays, out)
    sw = edition["statewide"]
    print(f"Model v1 edition as of {as_of}  draws={edition['draws']}  polls={edition['polls']['n']} "
          f"last fieldwork {edition['polls']['last_fieldwork']}")
    print("Poll level:", {p: round(100 * edition['polls']['level'][p], 1) for p in PARTIES})
    print("Statewide primary mean:", {p: round(100 * sw['primary'][p]['mean'], 1) for p in PARTIES})
    print(f"Statewide ALP 2PP (where defined): {100*sw['tpp_alp']['mean']:.1f} "
          f"[{100*sw['tpp_alp']['lo90']:.1f}, {100*sw['tpp_alp']['hi90']:.1f}]")
    print("Seats (mean, 90% interval, P majority):")
    for c, v in sw["seats"].items():
        print(f"  {c:4s} {v['mean']:5.1f}  [{v['lo90']:2d}, {v['hi90']:2d}]  P(maj)={v['p_majority']:.3f}")
    print("P(most seats):", {k: round(v, 3) for k, v in sw["p_most_seats"].items()}, " P(hung):", round(sw["p_hung"], 3))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
