"""
Fetch per-game schedule, dates, and final scores for every modeled season via
nba_api (LeagueGameLog). One game-log row per team per game is pivoted into a
single row per game with home/away teams and points.

Writes full_model/rapm/cache/games_{season}.csv:
    GAME_ID, DATE, SEASON, HOME, AWAY, HOME_PTS, AWAY_PTS, HOME_WIN

`season` follows the repo convention (end year): 2025 == the 2024-25 season.
Requires Python 3.10+ with nba_api installed (see ingest env note in plan).
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
CACHE.mkdir(exist_ok=True)

SEASONS = list(range(2015, 2027))   # 2014-15 .. 2025-26
SEASON_TYPES = ["Regular Season", "Playoffs"]
REQUEST_PAUSE = 0.6                 # be gentle with stats.nba.com

# normalize stats.nba.com abbreviations to the repo's convention
ABBR_FIX = {"BKN": "BRK", "PHX": "PHO", "CHA": "CHO"}


def season_str(end_year):
    return f"{end_year - 1}-{str(end_year)[2:]}"


def fetch_game_log(end_year, season_type):
    """All team game-log rows for a season + season type (one row per team-game)."""
    for attempt in range(4):
        try:
            res = leaguegamelog.LeagueGameLog(
                season=season_str(end_year),
                season_type_all_star=season_type,
                timeout=60,
            )
            return res.get_data_frames()[0]
        except Exception as exc:               # noqa: BLE001 - retry on any net error
            wait = 2 ** attempt
            print(f"  retry {attempt+1} ({season_type} {end_year}) after {wait}s: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {season_type} {end_year}")


def build_games(end_year):
    frames = []
    for st in SEASON_TYPES:
        df = fetch_game_log(end_year, st)
        if df is not None and len(df):
            df = df.assign(SEASON_TYPE=st)
            frames.append(df)
        time.sleep(REQUEST_PAUSE)
    if not frames:
        return pd.DataFrame()
    log = pd.concat(frames, ignore_index=True)

    # MATCHUP: "BOS vs. NYK" => BOS is home; "BOS @ NYK" => BOS is away.
    log["IS_HOME"] = log.MATCHUP.str.contains("vs.")
    home = log[log.IS_HOME].copy()
    away = log[~log.IS_HOME].copy()

    g = home.merge(
        away, on="GAME_ID", suffixes=("_H", "_A"), how="inner"
    )
    out = pd.DataFrame({
        "GAME_ID": g.GAME_ID.astype("int64"),
        "DATE": pd.to_datetime(g.GAME_DATE_H).dt.strftime("%Y-%m-%d"),
        "SEASON": end_year,
        "HOME": g.TEAM_ABBREVIATION_H,
        "AWAY": g.TEAM_ABBREVIATION_A,
        "HOME_PTS": g.PTS_H.astype(int),
        "AWAY_PTS": g.PTS_A.astype(int),
        "SEASON_TYPE": g.SEASON_TYPE_H,
    })
    out["HOME"] = out.HOME.replace(ABBR_FIX)
    out["AWAY"] = out.AWAY.replace(ABBR_FIX)
    out["HOME_WIN"] = (out.HOME_PTS > out.AWAY_PTS).astype(int)
    out = out.drop_duplicates("GAME_ID").sort_values("DATE").reset_index(drop=True)
    return out


def main():
    for y in SEASONS:
        games = build_games(y)
        if games.empty:
            print(f"season {y}: no games returned (skipped)")
            continue
        path = CACHE / f"games_{y}.csv"
        games.to_csv(path, index=False)
        reg = (games.SEASON_TYPE == "Regular Season").sum()
        print(f"season {y}: {len(games)} games ({reg} regular) "
              f"{games.DATE.min()}..{games.DATE.max()} -> {path.name}")


if __name__ == "__main__":
    main()
