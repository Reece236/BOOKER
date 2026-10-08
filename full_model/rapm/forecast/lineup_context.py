"""
Lineup-context stats from possession-level stints -- who a player shared the floor
with and against, plus his real on-court margin.

For each (season, player), possession-weighted over every stint he's on court:
  opp_quality  = average OPPONENT rating on the floor (impact/100) -> competition faced
  tm_quality   = average TEAMMATE rating on the floor (impact/100) -> supporting cast
  real_pm      = his actual on-court net rating /100  (real +/-)

These are DESCRIPTIVE context only. For a player's *isolated* impact use the regularized
BookerFormer impact_total -- a model-based expected_net / net_impact proxy was tried and
dropped: subtracting a regularized (and ~32% over-dispersed, slope 0.76) expectation from
the raw on-court net systematically inflated it (e.g. Wembanyama net_impact +20.7 vs a true
on/off of +7.9, which the regularized impact already nails at +7.9).

Player rating used for tm/opp quality = BookerFormer impact_total (same season; 0 for unrated).
Output cache/lineup_context.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
OUT = CACHE / "lineup_context.csv"
MIN_POSS = 200


def build(seasons=range(2018, 2027)):
    from . import player_impacts as pi
    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    imp = {(int(s), int(p)): float(v) for s, p, v in zip(rat.season, rat.PLAYER_ID, rat.impact_total)}
    data = pi.BookerData(seasons=range(min(seasons), max(seasons) + 1))
    rows = []
    for s in seasons:
        d = data.STINTS.get(s)
        if d is None:
            continue
        acc = {}   # pid -> [poss, net*poss, tmq*poss, oppq*poss]
        for h5, a5, poss, y in zip(d.home, d.away, d.POSS, d.Y):
            if poss <= 0 or not h5 or not a5:
                continue
            hi = [imp.get((s, int(p)), 0.0) for p in h5]
            ai = [imp.get((s, int(p)), 0.0) for p in a5]
            sh, sa = sum(hi), sum(ai)
            for k, p in enumerate(h5):
                a = acc.setdefault(int(p), [0.0] * 4)
                a[0] += poss; a[1] += y * poss
                a[2] += ((sh - hi[k]) / 4.0) * poss        # avg teammate impact
                a[3] += (sa / 5.0) * poss                  # avg opponent impact
            for k, p in enumerate(a5):
                a = acc.setdefault(int(p), [0.0] * 4)
                a[0] += poss; a[1] += (-y) * poss
                a[2] += ((sa - ai[k]) / 4.0) * poss
                a[3] += (sh / 5.0) * poss
        for p, a in acc.items():
            if a[0] < MIN_POSS:
                continue
            real, tmq, oppq = a[1]/a[0], a[2]/a[0], a[3]/a[0]
            rows.append({"season": int(s), "PLAYER_ID": int(p), "poss": round(a[0]),
                         "opp_quality": round(oppq, 2), "tm_quality": round(tmq, 2),
                         "real_pm": round(real, 2)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} player-seasons, {out.season.min()}-{out.season.max()})")
    return out


if __name__ == "__main__":
    build()
