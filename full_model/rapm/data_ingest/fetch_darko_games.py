"""Fetch DARKO's game-by-game DPM history per player from darko.app.

Source: https://darko.app/player/{nba_id}/__data.json (SvelteKit `devalue` payload).
The `history` table has one row per team game the player was on the roster for
(DNPs included, seconds_played=0) with dpm / o_dpm / d_dpm / box_dpm / on_off_dpm and
DARKO's own minutes projection x_minutes.

TIMING (verified empirically, 2022-26, n=111,880 player-games): a game row's dpm is
the PRE-GAME rating. The row-over-row change correlates with the player's own on-court
+/- in the PREVIOUS game (r=.262) and not with that game (r=.006) or the next (.002).
So the row value is leak-free for predicting that game; darko_pregame() returns it as
`*_pre` (lagging it again would make DARKO one game stale and handicap it).

Output:
    cache/darko_games/{nba_id}.parquet    raw per-player cache (resumable)
    cache/darko_games.parquet             combined (nba_id, date, season, seconds_played,
                                          tm_id, opp_id, dpm, o_dpm, d_dpm, x_minutes, ...)
Run:  python data_ingest/fetch_darko_games.py [--min-season 2016] [--pause 0.8]
Be polite: single-threaded, cached, ~1 request/sec.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
OUT_DIR = CACHE / "darko_games"
OUT_DIR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (BOOKER research; cached, ~1 req/s)"}
KEEP = ["nba_id", "date", "season", "seconds_played", "future_game", "tm_id", "opp_id",
        "dpm", "o_dpm", "d_dpm", "box_dpm", "on_off_dpm", "x_minutes", "x_pace", "age"]


def _decode(arr):
    memo = {}

    def h(i):
        if isinstance(i, int) and i < 0:
            return None
        if i in memo:
            return memo[i]
        v = arr[i]
        if isinstance(v, dict):
            out = {}
            memo[i] = out
            for k, j in v.items():
                out[k] = h(j)
            return out
        if isinstance(v, list):
            out = []
            memo[i] = out
            out.extend(h(j) for j in v)
            return out
        return v
    return h(0)


def fetch_player(pid, pause=0.8):
    path = OUT_DIR / f"{pid}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    url = f"https://www.darko.app/player/{pid}/__data.json"
    for attempt in range(3):
        try:
            raw = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
            break
        except Exception as exc:            # noqa: BLE001
            if "404" in str(exc):
                pd.DataFrame(columns=KEEP).to_parquet(path)     # cache the miss
                return None
            time.sleep(2 ** attempt * 2)
    else:
        return None
    time.sleep(pause)
    h = None
    for n in raw.get("nodes", []):
        if n and n.get("type") == "data":
            arr = n["data"]
            root = arr[0] if arr else None
            if isinstance(root, dict) and "history" in root:
                h = _decode(arr)["history"]          # decode once
                break
    if h is None:
        pd.DataFrame(columns=KEEP).to_parquet(path)
        return None
    df = pd.DataFrame(dict(zip(h["keys"], h["values"])))
    df = df[[c for c in KEEP if c in df.columns]]
    df.to_parquet(path)
    return df


def player_ids(min_season):
    ids = set()
    for p in CACHE.glob("players_*.csv"):
        try:
            s = int(p.stem.split("_")[1])
        except ValueError:
            continue
        if s >= min_season:
            d = pd.read_csv(p)
            ids |= set(d[d.MINUTES >= 100].PLAYER_ID.astype(int))
    return sorted(ids)


def combine():
    frames = [pd.read_parquet(p) for p in sorted(OUT_DIR.glob("*.parquet"))]
    frames = [f for f in frames if len(f)]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.future_game == 0].drop(columns="future_game")
    df.to_parquet(CACHE / "darko_games.parquet")
    print(f"wrote cache/darko_games.parquet ({len(df)} player-games, {df.nba_id.nunique()} players)")
    return df


def darko_pregame(df=None):
    """Leak-free pre-game DARKO (row values are already pre-game; see TIMING)."""
    df = pd.read_parquet(CACHE / "darko_games.parquet") if df is None else df.copy()
    df = df.sort_values(["nba_id", "date"])
    for c in ("dpm", "o_dpm", "d_dpm", "x_minutes"):
        df[c + "_pre"] = df[c]
    df["date"] = df.date.astype(str)
    return df


if __name__ == "__main__":
    args = sys.argv[1:]
    min_season = int(args[args.index("--min-season") + 1]) if "--min-season" in args else 2016
    pause = float(args[args.index("--pause") + 1]) if "--pause" in args else 0.8
    ids = player_ids(min_season)
    print(f"{len(ids)} players to fetch (cached ones skipped)")
    for i, pid in enumerate(ids, 1):
        fetch_player(pid, pause)
        if i % 100 == 0:
            print(f"  {i}/{len(ids)}", flush=True)
        elif i == 1:
            print("  started", flush=True)
    combine()
