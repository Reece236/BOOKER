"""
Build enhanced player ratings (total + offensive + defensive WAA) for export.

Uses teammate-fit adjusted impacts from forecast/enhanced_impacts.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import enhanced_impacts as ei
from forecast import player_impacts as pi

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SEASONS = list(range(2015, 2028))


def season_enhanced_rating(data, season):
    train = [s for s in range(season - 3, season) if s >= 2015 and s in data.STINTS]
    if season not in data.PLAYERS or not train:
        return None
    enh = ei.build_enhanced(data, train, season)
    k, _ = pi.fit_net_to_wins(data, train)
    rows = ei.player_waa_components(data, season, k, enh)
    if not rows:
        return None
    pl = pd.DataFrame(rows)
    pl["season"] = season
    pl = pl[pl.minutes >= 250].copy()
    pl["rank"] = pl.waa_total.rank(ascending=False, method="first").astype(int)
    pl["rank_off"] = pl.waa_off.rank(ascending=False, method="first").astype(int)
    pl["rank_def"] = pl.waa_def.rank(ascending=False, method="first").astype(int)
    return pl


def build_all_enhanced():
    data = pi.BookerData(seasons=SEASONS)
    parts = []
    for s in SEASONS:
        if s not in data.PLAYERS or not (CACHE / f"stints_{s}.csv").exists():
            continue
        df = season_enhanced_rating(data, s)
        if df is not None:
            parts.append(df)
            print(f"season {s}: {len(df)} players (enhanced O/D WAA)")
    if not parts:
        return pd.DataFrame()
    allr = pd.concat(parts, ignore_index=True)
    out = HERE / "booker_waa_enhanced_ratings.csv"
    allr.round({
        "minutes": 0, "impact_off": 2, "impact_def": 2, "impact_total": 2,
        "waa_off": 2, "waa_def": 2, "waa_total": 2,
    }).to_csv(out, index=False)
    print(f"wrote {out} ({len(allr)} rows)")
    return allr


if __name__ == "__main__":
    build_all_enhanced()
