"""Leverage index (LI) per stint and the leverage x player-type tests.

LI = d WinProb / d margin at the stint's start, normalized so the possession-weighted
league average is 1.0. Win prob: P(home) = Phi((m + mu*tau) / (sigma*sqrt(tau))), m =
running margin (real per-side points), tau = fraction of regulation remaining (OT uses
its own clock), mu = pre-game expected home margin (sequential filter + HCA), sigma =
full-game margin SD. Garbage time -> LI ~ 0 by construction; a tie game with two
minutes left -> LI ~ 5-8.

Output cache/stint_leverage.parquet: season, GAME_ID, stint index, LI, margin, tau.
Run: python -m forecast.leverage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import player_impacts as pi

CACHE = pi.CACHE
# Fitted by MLE on 2019-22 stint-start states vs final results (calibration-checked
# on 2023-26): a constant-variance Brownian margin with the full-game SD is
# overconfident late (fouling, threes, end-game variance) -> sigma x1.69; the pre-game
# net-rating edge is re-weighted to keep tip-off win probabilities calibrated.
SIGMA = pi.GAME_MARGIN_SD * 1.694
PREGAME_W = 1.732
HCA_WP = 3.998
TAU_MIN = 15.0 / 2880.0     # floor: last-15-second possessions don't explode
LI_CAP = 12.0


def season_leverage(s):
    st = pd.read_csv(CACHE / f"stints_{s}.csv")
    st = st[st.GAME_ID.astype(str).str.startswith("2")].copy()        # regular season
    st["row"] = np.arange(len(st))
    if "START_SEC" not in st.columns:
        # v3-built seasons (2026) don't store clocks; stints are contiguous within a
        # period in file order, so the start clock = period length - elapsed duration
        plen = np.where(st.PERIOD <= 4, 720.0, 300.0)
        elapsed = st.groupby(["GAME_ID", "PERIOD"]).DURATION_SECONDS.cumsum() - st.DURATION_SECONDS
        st["START_SEC"] = np.maximum(plen - elapsed, 1.0)
    st = st.sort_values(["GAME_ID", "PERIOD", "START_SEC"], ascending=[True, True, False])
    pm = (st.HOME_PTS - st.AWAY_PTS).astype(float)
    st["margin"] = pm.groupby(st.GAME_ID).cumsum() - pm               # margin BEFORE the stint
    reg = st.PERIOD <= 4
    rem = np.where(reg, (4 - st.PERIOD) * 720.0 + st.START_SEC, st.START_SEC)
    st["tau"] = np.maximum(rem / 2880.0, TAU_MIN)
    mu = pd.Series(HCA_WP, index=st.index)
    sp = CACHE / f"seq_games_{s}.csv"
    if sp.exists():                    # pre-game strength: leak-free sequential margin
        g = pd.read_csv(sp)
        exp = dict(zip(g.GAME_ID, PREGAME_W * (g.net_home_strict - g.net_away_strict) + HCA_WP))
        mu = st.GAME_ID.map(exp).fillna(HCA_WP)
    sd = SIGMA * np.sqrt(st.tau)
    z = (st.margin + mu * st.tau) / sd
    st["wp"] = norm.cdf(z)
    st["li_raw"] = norm.pdf(z) / sd
    return st


def build(seasons=range(2018, 2027)):
    frames = [season_leverage(s).assign(season=s) for s in seasons]
    d = pd.concat(frames, ignore_index=True)
    scale = np.average(d.li_raw, weights=d.POSS)
    d["LI"] = (d.li_raw / scale).clip(upper=LI_CAP)
    out = d[["season", "GAME_ID", "row", "PERIOD", "START_SEC", "margin", "tau", "wp", "LI"]]
    out.to_parquet(CACHE / "stint_leverage.parquet")
    print(f"wrote stint_leverage.parquet ({len(out)} stints); "
          f"LI pct of poss >2: {np.average(d.LI > 2, weights=d.POSS):.3f}, <0.25: {np.average(d.LI < .25, weights=d.POSS):.3f}")
    return out


if __name__ == "__main__":
    build()
