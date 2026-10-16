"""
Replacement level MEASURED from competitive possessions (not from shrunk ratings).

For a player group G in season s, on stints with leverage LI >= 0.5 (not garbage time):
    100*(home pts - away pts)/poss - [sum home ratings - sum away ratings]_(non-G)
        = HCA + g * (n_G home - n_G away)
with the other players' BookerFormer ratings (row s) as fixed offsets; g = the group's pooled
level per 100 vs average. Pooling avoids the shrinkage bias of averaging their own ratings
(fringe players' table ratings sit ~2 pts/100 above what they produce).

Groups: 'fringe' (< 500 min), 'garbage' (>= 50% of minutes at LI < 0.15), 'fill' (played for
a team but not on its opening roster -- the population the minutes model's replacement fill
stands in for). Measured 2018-26: fringe -3.9 (se .14), garbage -7.4 (se 1.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import player_impacts as pi

CACHE = pi.CACHE
_L = {}


def _stints(s):
    st = pd.read_csv(CACHE / f"stints_{s}.csv")
    st = st[st.GAME_ID.astype(str).str.startswith("2")].reset_index(drop=True)
    st["row"] = np.arange(len(st))
    if "L" not in _L:
        _L["L"] = pd.read_parquet(CACHE / "stint_leverage.parquet")
    L = _L["L"]
    st = st.merge(L[L.season == s][["row", "LI"]], on="row", how="left")
    st["H"] = st.HOME_LINEUP.map(pi.parse_lineup)
    st["A"] = st.AWAY_LINEUP.map(pi.parse_lineup)
    return st


def groups(s, st=None, fill_roster=None):
    st = _stints(s) if st is None else st
    sec, gsec = {}, {}
    for h, a, du, l in zip(st.H, st.A, st.DURATION_SECONDS, st.LI):
        for p in h + a:
            sec[p] = sec.get(p, 0) + du
            gsec[p] = gsec.get(p, 0) + (du if l < 0.15 else 0)
    out = {"fringe": {p for p in sec if sec[p] / 60 < 500},
           "garbage": {p for p in sec if gsec[p] / sec[p] >= 0.5 and sec[p] / 60 >= 50}}
    if fill_roster is not None:
        out["fill"] = fill_roster
    return out, st


def measure(s, group, st, rat):
    r = dict(zip(rat[rat.season == s].PLAYER_ID, rat[rat.season == s].impact_total))
    y = (100 * (st.HOME_PTS - st.AWAY_PTS) / st.POSS.clip(lower=0.5)).values
    X, Y, W = [], [], []
    for i in np.where((st.LI >= 0.5).values)[0]:
        h, a = st.H[i], st.A[i]
        if any((p not in r) and (p not in group) for p in h + a):
            continue
        off = sum(r.get(p, 0) for p in h if p not in group) - sum(r.get(p, 0) for p in a if p not in group)
        X.append(sum(p in group for p in h) - sum(p in group for p in a)); Y.append(y[i] - off); W.append(st.POSS[i])
    X, Y, W = np.array(X, float), np.array(Y), np.array(W)
    D = np.c_[np.ones_like(X), X]
    A_ = (D * W[:, None]).T @ D
    beta = np.linalg.solve(A_, (D * W[:, None]).T @ Y)
    s2 = np.average((Y - D @ beta) ** 2, weights=W)
    return float(beta[1]), float(np.sqrt(s2 * np.linalg.inv(A_)[1, 1]))


def pooled(seasons, which="fringe", fill_rosters=None, rat=None):
    """Inverse-variance pooled g over `seasons` (walk-forward: pass seasons < s)."""
    rat = pd.read_csv(pi.RAPM / "booker_bookerformer_ratings.csv") if rat is None else rat
    gs, ws = [], []
    for s in seasons:
        if not (CACHE / f"stints_{s}.csv").exists() or not (rat.season == s).any():
            continue
        G, st = groups(s, fill_roster=(fill_rosters or {}).get(s))
        if which not in G or not G[which]:
            continue
        g, se = measure(s, G[which], st, rat)
        gs.append(g); ws.append(1 / se ** 2)
    return (float(np.dot(gs, ws) / np.sum(ws)), float(1 / np.sqrt(np.sum(ws)))) if gs else (np.nan, np.nan)
