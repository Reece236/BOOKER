"""
Role / usage value: what a player is worth if USED PROPERLY.

Skill curve (identified within-player from RAW efficiency): true-shooting% falls ~BETA
per +1% usage (tougher shots -- xfg also falls with usage). So scoring value
value(u) = u*(TS(u) - r) is a downward parabola with an interior optimum u*. A player
used ABOVE u* is an "empty-calorie" scorer worth more in a smaller, more selective role;
below it he should get the ball more.

  TS(u)    = TS0 - BETA*(u - u0)
  value(u) ~ k * u * (TS(u) - r)          # k calibrated so this ~ impact_off (pts/100)
  u*       = (TS0 + BETA*u0 - r) / (2*BETA)

r is self-calibrated so the median player's optimum equals his usage (u* is then a
peer-relative "is your efficiency good enough for your volume?" verdict). BETA is the
within-player estimate -- conservative, since skill-development confounding attenuates it,
so the reported upside is a floor.

Output cache/role_value.csv per (season, PLAYER_ID): usage, opt_usage, misuse (usage-opt;
+ = over-used), ts_now, ts_opt, off_now, off_opt, role_upside (pts/100 offense gained if
used at his optimum).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from . import player_impacts as pi

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
OUT = CACHE / "role_value.csv"
MASTER = pi.ROOT / "full_model" / "nba_master_dataset_with_archetypes.csv"
# TS drop per +1 usage%. VALIDATED via the shot-DIET channel (2026-07): within-player
# d(diet_xfg)/d(usage) = -0.00119/pt (SE .0002, sign-consistent in both directions),
# ~= -0.00137 in TS terms -- naive season-delta TS estimates are confounded (usage
# drops are CAUSED by decline: wrong-sign asymmetry), but the mix channel is clean.
# 0.00142 is right, and a floor (contested-make channel unmeasurable without tracking).
BETA = 0.00142
MIN_MIN = 600


def _usage_ts():
    m = pd.read_csv(MASTER, usecols=["playerName", "season", "usagePercent",
                                     "total_points", "total_fieldAttempts", "total_ftAttempts"])
    m["nm"] = m.playerName.map(pi.norm_name)
    m["season"] = pd.to_numeric(m.season, errors="coerce")
    m = m.dropna(subset=["season"])
    m["ts"] = m.total_points / (2 * (m.total_fieldAttempts + 0.44 * m.total_ftAttempts))
    d = {(nm, int(s)): (float(u), float(t)) for nm, s, u, t in
         zip(m.nm, m.season, m.usagePercent, m.ts) if pd.notna(u) and pd.notna(t) and t > 0}
    latest = {}
    for (nm, s), v in d.items():
        if nm not in latest or s > latest[nm][0]:
            latest[nm] = (s, v)
    return d, latest


def build(seasons=range(2018, 2027)):
    d, latest = _usage_ts()
    rat = pd.read_csv(HERE.parent / "booker_bookerformer_ratings.csv")[
        ["PLAYER_ID", "season", "player", "minutes", "impact_off"]]
    rat = rat[rat.season.isin(seasons)].copy()
    rat["nm"] = rat.player.map(pi.norm_name)

    def uv(nm, s):
        if (nm, int(s)) in d:
            return d[(nm, int(s))]
        L = latest.get(nm)
        return L[1] if L else (np.nan, np.nan)
    got = [uv(nm, s) for nm, s in zip(rat.nm, rat.season)]
    rat["usage"] = [g[0] for g in got]
    rat["ts"] = [g[1] for g in got]
    X = rat[(rat.minutes >= MIN_MIN) & rat.usage.notna() & rat.ts.notna() & (rat.usage > 5)].copy()

    # self-calibrated replacement r (median player optimally used at his usage) + value scale k
    X["c"] = X.ts + BETA * X.usage
    r = float(np.median(X.c) - 2 * BETA * np.median(X.usage))
    lt = X.usage * (X.ts - r)
    k = float(np.sum(lt * X.impact_off) / np.sum(lt * lt))

    optU = np.clip((X.c - r) / (2 * BETA), 10.0, 34.0)
    ts_opt = X.ts + BETA * (X.usage - optU)
    off_now = k * X.usage * (X.ts - r)
    off_opt = k * optU * (ts_opt - r)
    out = pd.DataFrame({
        "season": X.season.astype(int), "PLAYER_ID": X.PLAYER_ID.astype(int),
        "usage": X.usage.round(1), "opt_usage": optU.round(1),
        "misuse": (X.usage - optU).round(1),
        "ts_now": X.ts.round(3), "ts_opt": ts_opt.round(3),
        "off_now": off_now.round(2), "off_opt": off_opt.round(2),
        "role_upside": (off_opt - off_now).round(2),
    })
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} player-seasons); r={r:.3f} k={k:.3f} beta={BETA}")
    return out


if __name__ == "__main__":
    build()
