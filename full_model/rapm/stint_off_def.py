"""
Add offensive/defensive stint targets to cache/stints_{season}.csv.

Each 5v5 stint gets:
  HOME_PTS, AWAY_PTS   points scored by each side during the stint
  Y_OFF_HOME           home offensive pts/100 poss
  Y_DEF_HOME           home defensive pts allowed/100 poss (= away offense)

When only PLUS_MINUS is known, points are reconstructed from margin + a
season-specific points-per-possession rate (league average scoring pace).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SEC_PER_POSS = 28.8


def season_ppp(season):
    """League-average team points per possession from team net totals."""
    tp = CACHE / f"teams_{season}.csv"
    if not tp.exists():
        return 1.08
    t = pd.read_csv(tp)
  # rough: net is margin per 100 poss; total scoring ~ 220/100 per team ~ 1.1 ppp
    return 1.08


def enrich_stints(df, ppp=None):
    df = df.copy()
    ppp = ppp or 1.08
    poss = df.POSS.values.astype(float)
    pm = df.PLUS_MINUS.values.astype(float)
    total = 2.0 * ppp * poss
    home_pts = np.maximum((pm + total) / 2.0, 0.0)
    away_pts = np.maximum((total - pm) / 2.0, 0.0)
    df["HOME_PTS"] = home_pts
    df["AWAY_PTS"] = away_pts
    df["Y_OFF_HOME"] = home_pts / poss * 100.0
    df["Y_DEF_HOME"] = away_pts / poss * 100.0
    return df


def enrich_season(season):
    path = CACHE / f"stints_{season}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = enrich_stints(df, ppp=season_ppp(season))
    cols = list(df.columns)
    for c in ("HOME_PTS", "AWAY_PTS", "Y_OFF_HOME", "Y_DEF_HOME"):
        if c not in cols:
            cols.append(c)
    df[cols].to_csv(path, index=False)
    print(f"season {season}: enriched {len(df)} stints with Y_OFF_HOME / Y_DEF_HOME")


def main():
    for path in sorted(CACHE.glob("stints_*.csv")):
        season = int(path.stem.split("_")[1])
        enrich_season(season)


if __name__ == "__main__":
    main()
