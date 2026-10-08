"""Builders for the two orphaned stint-derived artifacts every evaluation leans on:

  cache/pure_rapm.csv  season, PLAYER_ID, pure_rapm, minutes
      Single-season, PRIOR-FREE, possession-weighted ridge RAPM on net Y, alpha=2800.
      This is the "neutral arbiter" (next-season pure RAPM) in metric_eval.py and the
      BOOKER-PROJ debiased target / residual calibration.
  cache/h2_rapm.csv    season, PLAYER_ID, h2_impact
      SECOND-HALF-of-season form: ridge (alpha=2800) on regular-season stints dated on
      or after the season's median game date, centered on that season's box prior
      (player_impacts._decayed_priors(data, [s], s+1)). PROJ's recency feature.

Provenance (2026-07 audit): the original builders lived only in an ephemeral agent
scratchpad and were lost. The specs above were RECOVERED empirically by refitting
candidate specifications on the pre-audit stints and matching the existing CSVs:
pure_rapm r=.9988 / slope 0.998, h2_rapm r=.9986 / slope 0.994 (2019). Both must be
rebuilt whenever stints change -- the old files were computed on the buggy stints
(see data_ingest/backfill_stint_points.py).

Run:  python -m forecast.arbiters [seasons...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from . import player_impacts as pi

CACHE = pi.CACHE
ALPHA = 2800.0
MIN_MINUTES = 250


def _design(stints):
    ids = sorted({p for l in stints.home for p in l} | {p for l in stints.away for p in l})
    col = {p: i for i, p in enumerate(ids)}
    r, c, v = [], [], []
    for i, (h, a) in enumerate(zip(stints.home, stints.away)):
        for p in h:
            r.append(i); c.append(col[p]); v.append(1.0)
        for p in a:
            r.append(i); c.append(col[p]); v.append(-1.0)
    return csr_matrix((v, (r, c)), shape=(len(stints), len(ids))), ids


def _regular(data, s):
    d = data.STINTS[s]
    return d[d.GAME_ID.astype(str).str.startswith("2")]


def pure_rapm(data, s):
    d = _regular(data, s)
    X, ids = _design(d)
    m = Ridge(alpha=ALPHA).fit(X, d.Y, sample_weight=d.POSS)
    return dict(zip(ids, m.coef_))


def h2_rapm(data, s):
    d = _regular(data, s)
    g = data.GAMES[s]
    g = g[g.SEASON_TYPE == "Regular Season"]
    mid = g.DATE.sort_values().iloc[len(g) // 2]
    dates = dict(zip(g.GAME_ID.astype("int64"), g.DATE))
    h2 = d[d.GAME_ID.astype("int64").map(dates) >= mid]
    X, ids = _design(h2)
    prior, _ = pi._decayed_priors(data, [s], s + 1)
    b0 = np.array([prior.get(p, pi.PRIOR_BASE) for p in ids])
    m = Ridge(alpha=ALPHA).fit(X, h2.Y - X.dot(b0), sample_weight=h2.POSS)
    return dict(zip(ids, b0 + m.coef_))


def build(seasons=range(2018, 2027)):
    data = pi.BookerData(seasons=range(min(seasons) - 1, max(seasons) + 2))
    pr_rows, h2_rows = [], []
    for s in seasons:
        if s not in data.STINTS or s not in data.PLAYERS or s not in data.GAMES:
            continue
        mins = dict(zip(data.PLAYERS[s].PLAYER_ID, data.PLAYERS[s].MINUTES))
        pr, h2 = pure_rapm(data, s), h2_rapm(data, s)
        for p, v in pr.items():
            if mins.get(p, 0) >= MIN_MINUTES:
                pr_rows.append({"season": s, "PLAYER_ID": int(p), "pure_rapm": round(float(v), 2),
                                "minutes": int(round(mins[p]))})
        for p, v in h2.items():
            if mins.get(p, 0) >= MIN_MINUTES:
                h2_rows.append({"season": s, "PLAYER_ID": int(p), "h2_impact": round(float(v), 2)})
        print(f"  {s}: pure_rapm {sum(r['season'] == s for r in pr_rows)}, "
              f"h2 {sum(r['season'] == s for r in h2_rows)}")

    def _merge(path, new):
        new = pd.DataFrame(new)
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old.season.isin(set(new.season))]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(["season", "PLAYER_ID"]).to_csv(path, index=False)
        print(f"wrote {path} ({len(new)} rows)")

    _merge(CACHE / "pure_rapm.csv", pr_rows)
    _merge(CACHE / "h2_rapm.csv", h2_rows)


if __name__ == "__main__":
    ss = [int(a) for a in sys.argv[1:]]
    build(ss or range(2018, 2027))
