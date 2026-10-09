"""End-of-season incentive state per team-game, as of the morning of the game.

Standings are rebuilt day by day from completed regular-season results (no lookahead).
Per team before each game:
  games_left    regular-season games remaining
  elim          mathematically out of the play-in (wins + games_left < current 10th-place
                wins in the conference)
  locked        seed effectively settled: gap to the seed above AND below both exceed
                games_left (nothing left to play for; rest risk)
  tank          bottom-4 league record with <= TANK_WINDOW games left (lottery odds)
  low_motiv     elim | locked | tank, only in the final LATE_GAMES games
Feature for game odds: motiv_diff = low_motiv(away) - low_motiv(home).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import player_impacts as pi

LATE_GAMES = 20
TANK_WINDOW = 20


def season_incentives(games):
    g = games[games.SEASON_TYPE == "Regular Season"].sort_values(["DATE", "GAME_ID"])
    teams = sorted(set(g.HOME) | set(g.AWAY))
    total = pd.concat([g.HOME, g.AWAY]).value_counts().to_dict()
    w = {t: 0 for t in teams}; l = {t: 0 for t in teams}
    out = {}
    for date, day in g.groupby("DATE", sort=True):
        gp = {t: w[t] + l[t] for t in teams}
        left = {t: total[t] - gp[t] for t in teams}
        state = {}
        for conf in ("E", "W"):
            ct = [t for t in teams if pi.CONFERENCE.get(t) == conf]
            order = sorted(ct, key=lambda t: (w[t] - l[t]), reverse=True)
            tenth_w = w[order[9]] if len(order) >= 10 else 0
            for i, t in enumerate(order):
                up = (w[order[i - 1]] - w[t]) if i > 0 else 99
                dn = (w[t] - w[order[i + 1]]) if i + 1 < len(order) else 99
                state[t] = {"elim": int(w[t] + left[t] < tenth_w),
                            "locked": int(min(up, dn) > left[t] and left[t] < LATE_GAMES)}
        league = sorted(teams, key=lambda t: w[t] - l[t])
        bottom4 = set(league[:4])
        for t in teams:
            late = left[t] <= LATE_GAMES
            s = state[t]
            s["tank"] = int(t in bottom4 and left[t] <= TANK_WINDOW)
            s["low_motiv"] = int(late and (s["elim"] or s["locked"] or s["tank"]))
            s["games_left"] = left[t]
        for r in day.itertuples():
            out[int(r.GAME_ID)] = {"home": dict(state[r.HOME]), "away": dict(state[r.AWAY])}
        for r in day.itertuples():
            if pd.isna(r.HOME_WIN):
                continue
            if int(r.HOME_WIN) == 1:
                w[r.HOME] += 1; l[r.AWAY] += 1
            else:
                w[r.AWAY] += 1; l[r.HOME] += 1
    return out
