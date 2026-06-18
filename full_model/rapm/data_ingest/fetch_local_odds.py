"""
Ingest local NBA betting-line CSVs into cache/odds_{season}.csv.

Sources (full_model/):
  * oddsData.csv          — American ML + spread, 2008–2023 (partial 2023)
  * nba_main_lines.csv    — Pinnacle decimal snapshots, 2025–26 season
  * nba_detailed_odds.csv — granular Pinnacle markets (optional fallback)

Output columns match fetch_odds / fetch_vegaslines:
    DATE, HOME, AWAY, ML_HOME, ML_AWAY, P_HOME, P_AWAY, HOME_SPREAD, OVER_TOTAL, SOURCE

Merge priority per game (newest / best source wins):
    pinnacle > oddsdata > existing cache (SBR)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
ROOT = HERE.parent.parent

ODDS_DATA_PATH = ROOT / "oddsData.csv"
MAIN_PATH = ROOT / "nba_main_lines.csv"
DETAIL_PATH = ROOT / "nba_detailed_odds.csv"

FULL_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

ODDS_DATA_TEAM = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BRK", "New Jersey": "BRK",
    "Charlotte": "CHO", "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL",
    "Denver": "DEN", "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU",
    "Indiana": "IND", "LA Clippers": "LAC", "LA Lakers": "LAL", "Memphis": "MEM",
    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP",
    "New York": "NYK", "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI",
    "Phoenix": "PHO", "Portland": "POR", "Sacramento": "SAC", "San Antonio": "SAS",
    "Toronto": "TOR", "Utah": "UTA", "Washington": "WAS",
}

SKIP_TEAMS = {"USA Stars", "USA Stripes", "World"}
OUT_COLS = [
    "DATE", "HOME", "AWAY", "ML_HOME", "ML_AWAY", "P_HOME", "P_AWAY",
    "HOME_SPREAD", "OVER_TOTAL", "SOURCE",
]
SOURCE_RANK = {"pinnacle": 3, "oddsdata": 2, "sbr_ml": 1, "sbr": 1, "spread": 1}


def ml_to_prob(ml):
    ml = float(ml)
    if ml < 0:
        return -ml / (-ml + 100.0)
    return 100.0 / (ml + 100.0)


def dec_to_american(dec):
    d = float(dec)
    if d <= 1.0 or np.isnan(d):
        return np.nan
    if d >= 2.0:
        return int(round((d - 1) * 100))
    return int(round(-100 / (d - 1)))


def dec_implied_prob(dec):
    d = float(dec)
    if d <= 1.0 or np.isnan(d):
        return np.nan
    return 1.0 / d


def _row_from_ml(date, home, away, ml_h, ml_a, spread=np.nan, total=np.nan, source="oddsdata"):
    try:
        ml_h, ml_a = int(ml_h), int(ml_a)
    except (TypeError, ValueError):
        return None
    ph, pa = ml_to_prob(ml_h), ml_to_prob(ml_a)
    s = ph + pa
    return {
        "DATE": str(date)[:10],
        "HOME": home,
        "AWAY": away,
        "ML_HOME": ml_h,
        "ML_AWAY": ml_a,
        "P_HOME": round(ph / s, 4),
        "P_AWAY": round(pa / s, 4),
        "HOME_SPREAD": spread,
        "OVER_TOTAL": total,
        "SOURCE": source,
    }


def load_odds_data():
    if not ODDS_DATA_PATH.exists():
        print(f"skip oddsData: {ODDS_DATA_PATH.name} not found")
        return pd.DataFrame(columns=OUT_COLS)
    df = pd.read_csv(ODDS_DATA_PATH)
    home = df[df["home/visitor"] == "vs"].copy()
    home["HOME"] = home.team.map(ODDS_DATA_TEAM)
    home["AWAY"] = home.opponent.map(ODDS_DATA_TEAM)
    home = home.dropna(subset=["HOME", "AWAY"])
    rows = []
    for r in home.itertuples():
        row = _row_from_ml(
            r.date, r.HOME, r.AWAY, r.moneyLine, r.opponentMoneyLine,
            spread=r.spread, total=r.total, source="oddsdata",
        )
        if row:
            rows.append(row)
    out = pd.DataFrame(rows)
    print(f"oddsData: {len(out)} home-side games ({ODDS_DATA_PATH.name})")
    return out


def load_main_lines():
    if not MAIN_PATH.exists():
        print(f"skip pinnacle main: {MAIN_PATH.name} not found")
        return pd.DataFrame()
    df = pd.read_csv(MAIN_PATH)
    df["ts"] = pd.to_datetime(df.timestamp)
    df["t1"] = df.team1.map(FULL_NAME_TO_ABBR)
    df["t2"] = df.team2.map(FULL_NAME_TO_ABBR)
    df = df[~df.team1.isin(SKIP_TEAMS)].dropna(subset=["t1", "t2"])
    return df


def _pair_key(a, b):
    return tuple(sorted([a, b]))


def _row_from_pinnacle(row, home, away):
    if row.t1 == home:
        ml_h, ml_a = row.team1_moneyline, row.team2_moneyline
        spread, total = row.team1_spread, row.over_total
    else:
        ml_h, ml_a = row.team2_moneyline, row.team1_moneyline
        spread, total = row.team2_spread, row.over_total
    ml_h, ml_a = dec_to_american(ml_h), dec_to_american(ml_a)
    return _row_from_ml(
        row.ts, home, away, ml_h, ml_a, spread=spread, total=total, source="pinnacle",
    )


def match_pinnacle_season(main_df, season, max_day_gap=30):
    sched_path = CACHE / f"games_{season}.csv"
    if not sched_path.exists() or main_df.empty:
        return pd.DataFrame(columns=OUT_COLS)
    sched = pd.read_csv(sched_path)
    sched = sched[sched.get("SEASON_TYPE", "Regular Season") == "Regular Season"]
    sched = sched.sort_values("DATE")

    links = main_df.sort_values("ts").groupby("game_link", as_index=False).last()
    by_pair = {}
    for r in links.itertuples():
        by_pair.setdefault(_pair_key(r.t1, r.t2), []).append(r)

    used_links = set()
    rows = []
    for g in sched.itertuples():
        pk = _pair_key(g.HOME, g.AWAY)
        cands = [c for c in by_pair.get(pk, []) if c.game_link not in used_links]
        if not cands:
            continue
        gdate = pd.Timestamp(g.DATE).normalize()
        scored = [
            (abs((c.ts.normalize() - gdate).days), c)
            for c in cands
            if abs((c.ts.normalize() - gdate).days) <= max_day_gap
        ]
        if not scored:
            continue
        scored.sort(key=lambda x: (x[0], -x[1].ts.timestamp()))
        pick = scored[0][1]
        payload = _row_from_pinnacle(pick, g.HOME, g.AWAY)
        if payload is None:
            continue
        payload["DATE"] = str(g.DATE)[:10]
        used_links.add(pick.game_link)
        rows.append(payload)
    return pd.DataFrame(rows)


def load_pinnacle_all():
    main = load_main_lines()
    if main.empty:
        return pd.DataFrame(columns=OUT_COLS)
    seasons = sorted({
        int(p.stem.split("_")[1])
        for p in CACHE.glob("games_*.csv")
        if p.stem.split("_")[1].isdigit() and int(p.stem.split("_")[1]) >= 2023
    })
    parts = [match_pinnacle_season(main, s) for s in seasons]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=OUT_COLS)
    out = pd.concat(parts, ignore_index=True)
    print(f"pinnacle: {len(out)} schedule-matched games ({MAIN_PATH.name})")
    return out


def _rank(source):
    return SOURCE_RANK.get(str(source), 0)


def combine_sources(local_df):
    """Merge local rows; pinnacle beats oddsdata on duplicate keys."""
    if local_df.empty:
        return local_df
    local_df = local_df.copy()
    local_df["_rank"] = local_df.SOURCE.map(_rank)
    local_df = local_df.sort_values("_rank", ascending=False)
    local_df = local_df.drop_duplicates(subset=["DATE", "HOME", "AWAY"], keep="first")
    return local_df.drop(columns=["_rank"], errors="ignore")


def merge_with_cache(season, local_df):
    out_path = CACHE / f"odds_{season}.csv"
    local = local_df[local_df.DATE.map(lambda d: _season_of(str(d)) == season)].copy()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        for c in OUT_COLS:
            if c not in existing.columns:
                existing[c] = np.nan
        if local.empty:
            existing[OUT_COLS].sort_values(["DATE", "HOME", "AWAY"]).to_csv(out_path, index=False)
            return len(existing)
        ex = existing.copy()
        ex["_rank"] = ex.SOURCE.fillna("sbr").map(_rank)
        combined = pd.concat([local.assign(_rank=local.SOURCE.map(_rank)), ex], ignore_index=True)
        combined = combined.sort_values("_rank", ascending=False)
        combined = combined.drop_duplicates(subset=["DATE", "HOME", "AWAY"], keep="first")
        merged = combined.drop(columns=["_rank"], errors="ignore").sort_values(["DATE", "HOME", "AWAY"])
    else:
        merged = local.sort_values(["DATE", "HOME", "AWAY"]) if not local.empty else pd.DataFrame(columns=OUT_COLS)
    merged[OUT_COLS].to_csv(out_path, index=False)
    n_local = len(local)
    print(f"season {season}: {n_local} local lines -> {out_path.name} ({len(merged)} total)")
    return len(merged)


def _season_of(date_str):
    y, m = int(date_str[:4]), int(date_str[5:7])
    return y + 1 if m >= 8 else y


def main():
    oddsdata = load_odds_data()
    pinnacle = load_pinnacle_all()
    local = combine_sources(pd.concat([oddsdata, pinnacle], ignore_index=True))
    if local.empty:
        print("no local odds ingested")
        return

    seasons = sorted({ _season_of(d) for d in local.DATE })
    seasons = [s for s in seasons if 2015 <= s <= 2027]
    for s in seasons:
        merge_with_cache(s, local)

    # also refresh pinnacle-only seasons even if oddsdata empty for that year
    for s in sorted({ _season_of(d) for d in pinnacle.DATE }) if not pinnacle.empty else []:
        if s not in seasons:
            merge_with_cache(s, local)


if __name__ == "__main__":
    main()
