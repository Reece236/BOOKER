"""
Download bulk shot-chart detail (one file per season) from shufinskiy/nba_data via
nba_on_court, for the BookerFormer shot-quality model.

shufinskiy years are (our season) - 1, matching build_all_stints.py:
    shuf 2014  ==  2014-15  ==  our season 2015
Saves cache/shotdetail_<our_season>.csv with one row per shot: SHOT_DISTANCE,
SHOT_ZONE_*, SHOT_TYPE (2PT/3PT), ACTION_TYPE (Pullup / Step Back / Jump Shot /
Driving Layup ...), SHOT_MADE_FLAG, PLAYER_ID, GAME_ID, period/clock.

Usage:
    python data_ingest/fetch_shotdetail.py            # our seasons 2015..2026
    python data_ingest/fetch_shotdetail.py 2025 2026  # specific our-seasons
"""
import shutil
import sys
import time
from pathlib import Path

import nba_on_court as noc

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
TMP = CACHE / "_shottmp"


def fetch(our_seasons):
    CACHE.mkdir(parents=True, exist_ok=True)
    for our in our_seasons:
        shuf = our - 1
        out = CACHE / f"shotdetail_{our}.csv"
        if out.exists():
            print(f"{our}: exists ({out.stat().st_size//1024} KB), skip")
            continue
        if TMP.exists():
            shutil.rmtree(TMP)
        TMP.mkdir(parents=True)
        try:
            t = time.time()
            noc.load_nba_data(path=TMP, seasons=[shuf], data="shotdetail",
                              seasontype="rg", untar=True)
            src = TMP / f"shotdetail_{shuf}.csv"
            if not src.exists():
                print(f"{our}: no shotdetail for shuf {shuf} (unavailable)")
                continue
            shutil.move(str(src), str(out))
            print(f"{our}: {out.stat().st_size//1024} KB ({time.time()-t:.0f}s)")
        except Exception as exc:
            print(f"{our}: FAILED -- {type(exc).__name__}: {exc}")
        finally:
            if TMP.exists():
                shutil.rmtree(TMP)


if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or list(range(2015, 2027))
    fetch(seasons)
    print("done")
