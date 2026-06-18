"""
Clone the 2026 regular-season schedule to 2027 for forward-looking forecasts.

Uses the same home/away matchups as 2025-26, shifts dates forward one calendar
year, and clones the end-of-season roster file as players_2027.csv.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
SRC_SEASON = 2026
DST_SEASON = 2027


def clone_games():
    src = CACHE / f"games_{SRC_SEASON}.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    g = pd.read_csv(src)
    out = g.copy()
    out["SEASON"] = DST_SEASON

    def shift_date(d):
        dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
        try:
            return (dt.replace(year=dt.year + 1)).strftime("%Y-%m-%d")
        except ValueError:
            # Feb 29
            return (dt + timedelta(days=365)).strftime("%Y-%m-%d")

    out["DATE"] = out.DATE.map(shift_date)
    # schedule only — drop realized scores/outcomes from the template season
    for col in ("HOME_PTS", "AWAY_PTS", "HOME_WIN"):
        if col in out.columns:
            out[col] = np.nan
    # synthetic game ids: prefix 227 + last 8 digits of source id
    out["GAME_ID"] = out.GAME_ID.astype(str).map(
        lambda x: int("227" + str(x)[-8:]) if len(str(x)) >= 8 else int("2270000000") + int(x) % 10000000
    )
    dst = CACHE / f"games_{DST_SEASON}.csv"
    out.to_csv(dst, index=False)
    reg = out[out.get("SEASON_TYPE", "Regular Season") == "Regular Season"]
    print(f"wrote {dst} ({len(reg)} regular-season games cloned from {SRC_SEASON})")
    return out


def clone_roster():
    src = CACHE / f"players_{SRC_SEASON}.csv"
    if not src.exists():
        raise FileNotFoundError(src)
    pl = pd.read_csv(src)
    dst = CACHE / f"players_{DST_SEASON}.csv"
    pl.to_csv(dst, index=False)
    print(f"wrote {dst} ({len(pl)} players cloned from {SRC_SEASON})")
    return pl


def clone_teams_placeholder():
    """Team actual net unknown for future season; copy structure with NaN nets."""
    src = CACHE / f"teams_{SRC_SEASON}.csv"
    if not src.exists():
        return
    t = pd.read_csv(src)
    t["ACTUAL_NET"] = float("nan")
    dst = CACHE / f"teams_{DST_SEASON}.csv"
    t.to_csv(dst, index=False)
    print(f"wrote {dst} (team shell for {DST_SEASON})")


def main():
    clone_games()
    clone_roster()
    clone_teams_placeholder()


if __name__ == "__main__":
    main()
