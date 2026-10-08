"""
Add offensive/defensive stint targets to cache/stints_{season}.csv.

Each 5v5 stint gets:
  HOME_PTS, AWAY_PTS   points scored by each side during the stint
  Y_OFF_HOME           home offensive pts/100 poss
  Y_DEF_HOME           home defensive pts allowed/100 poss (= away offense)
  PTS_SOURCE           "pbp" (real running score) or "synthetic"

Audit fix (2026-07). The old version ALWAYS rebuilt HOME_PTS/AWAY_PTS from
PLUS_MINUS at a constant 1.08 points/possession:
    home_pts = (pm + 2.16*poss)/2  ->  Y_OFF_HOME - 108 == +Y/2,  Y_DEF_HOME - 108 == -Y/2
so the "offense" and "defense" observations carried no information beyond net
margin. Summing a stint's two observations gives sum_{10 players}(off - def) = 0
with the precision of a real observation, which pins every player's off ~= def
(corr(impact_off, impact_def) was ~0.8 at every minutes tier). The O/D split was
therefore prior-driven, not measured.

Now: real per-side points (written by build_season_stints / fetch_nba_2026 or the
backfill in data_ingest/backfill_stint_points.py) are used whenever present. The
synthetic fallback is kept ONLY so old caches still load, is flagged
PTS_SOURCE="synthetic", and prints a loud warning -- an O/D split fit on it is not
identified. Downstream models should center offense targets with season_league_ppp100()
(the real league scoring level per 100 stint-possessions varies ~100 -> ~115 across
2015-2026), not a constant.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SEC_PER_POSS = 28.8
SYNTH_PPP = 1.08


def _has_real_points(df):
    if "HOME_PTS" not in df.columns or "AWAY_PTS" not in df.columns:
        return False
    if "PTS_SOURCE" in df.columns:
        return bool((df.PTS_SOURCE == "pbp").all())
    # legacy caches have synthetic HOME_PTS: they satisfy home-away == pm AND
    # home+away == 2*1.08*poss exactly. Real points are integers.
    hp = df.HOME_PTS.to_numpy(float); ap = df.AWAY_PTS.to_numpy(float)
    if np.isnan(hp).any() or np.isnan(ap).any():
        return False
    return bool(np.allclose(hp, np.round(hp)) and np.allclose(ap, np.round(ap)))


def enrich_stints(df, ppp=None):
    df = df.copy()
    poss = df.POSS.values.astype(float)
    if _has_real_points(df):
        home_pts = df.HOME_PTS.values.astype(float)
        away_pts = df.AWAY_PTS.values.astype(float)
        df["PTS_SOURCE"] = "pbp"
    else:
        ppp = ppp or SYNTH_PPP
        pm = df.PLUS_MINUS.values.astype(float)
        total = 2.0 * ppp * poss
        home_pts = np.maximum((pm + total) / 2.0, 0.0)
        away_pts = np.maximum((total - pm) / 2.0, 0.0)
        df["PTS_SOURCE"] = "synthetic"
    df["HOME_PTS"] = home_pts
    df["AWAY_PTS"] = away_pts
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Y_OFF_HOME"] = home_pts / poss * 100.0
        df["Y_DEF_HOME"] = away_pts / poss * 100.0
    return df


def season_league_ppp100(df):
    """Possession-weighted league offense, points per 100 stint-possessions."""
    if "HOME_PTS" not in df.columns:
        return 108.0
    pts = float(df.HOME_PTS.sum() + df.AWAY_PTS.sum())
    poss = float(2.0 * df.POSS.sum())
    return 100.0 * pts / poss if poss > 0 else 108.0


def enrich_season(season):
    path = CACHE / f"stints_{season}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = enrich_stints(df)
    cols = list(df.columns)
    for c in ("HOME_PTS", "AWAY_PTS", "Y_OFF_HOME", "Y_DEF_HOME", "PTS_SOURCE"):
        if c not in cols:
            cols.append(c)
    df[cols].to_csv(path, index=False)
    src = df.PTS_SOURCE.iloc[0] if len(df) else "?"
    msg = f"season {season}: enriched {len(df)} stints with Y_OFF_HOME / Y_DEF_HOME ({src} points)"
    if src != "pbp":
        msg += ("  ** WARNING: synthetic O/D targets -- the offense/defense split is NOT "
                "identified; run data_ingest/backfill_stint_points.py **")
    print(msg)


def main():
    for path in sorted(CACHE.glob("stints_*.csv")):
        try:
            season = int(path.stem.split("_")[1])
        except ValueError:          # stints_po_*.csv etc.
            continue
        enrich_season(season)


if __name__ == "__main__":
    main()
