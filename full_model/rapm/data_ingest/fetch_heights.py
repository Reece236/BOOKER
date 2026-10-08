"""One-time fetch of player height/weight from nba_api bulk bio endpoint.

LeagueDashPlayerBioStats returns every player's height (inches) + weight for a season
in a single call. Height is ~constant per player, so we pull all seasons and keep one
row per player (median height). Writes cache/player_heights.csv (PLAYER_ID, height_in,
weight, position-agnostic). Run once: `python -m data_ingest.fetch_heights`.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "cache"
OUT = CACHE / "player_heights.csv"


def build(seasons=range(2015, 2027)):
    from nba_api.stats.endpoints import leaguedashplayerbiostats as bio
    frames = []
    for s in seasons:
        tag = f"{s - 1}-{str(s)[2:]}"
        try:
            d = bio.LeagueDashPlayerBioStats(season=tag, timeout=30).get_data_frames()[0]
            d = d[["PLAYER_ID", "PLAYER_NAME", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"]]
            frames.append(d)
            print(f"  bio {tag}: {len(d)} players", flush=True)
        except Exception as e:
            print(f"  bio {tag} FAILED: {type(e).__name__} {str(e)[:60]}", flush=True)
        time.sleep(0.8)
    if not frames:
        return pd.DataFrame()
    allb = pd.concat(frames, ignore_index=True)
    allb["PLAYER_WEIGHT"] = pd.to_numeric(allb.PLAYER_WEIGHT, errors="coerce")
    g = allb.groupby("PLAYER_ID").agg(
        player=("PLAYER_NAME", "last"),
        height_in=("PLAYER_HEIGHT_INCHES", "median"),
        weight=("PLAYER_WEIGHT", "median")).reset_index()
    g = g[g.height_in.notna() & (g.height_in > 0)]
    g["height_in"] = g.height_in.round().astype(int)
    g.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(g)} players)")
    return g


if __name__ == "__main__":
    build()
