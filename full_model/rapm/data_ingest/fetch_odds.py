"""
Build per-season historical moneyline files with vig-removed implied win
probabilities, used to benchmark the BOOKER game-odds model against the market.

Primary source: a directly-fetchable pre-scraped Sportsbook Review archive
(closing moneylines per game), which covers repo seasons ~2015-2022. Recent seasons
(2023+) are filled from local CSVs via fetch_local_odds.py:
    full_model/oddsData.csv
    full_model/nba_main_lines.csv
    full_model/nba_detailed_odds.csv

Writes cache/odds_{season}.csv:
    DATE, HOME, AWAY, ML_HOME, ML_AWAY, P_HOME, P_AWAY
where P_* are normalized (no-vig) implied win probabilities.
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW = CACHE / "odds_archive.json"
SRC = ("https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/"
       "main/data/nba_archive_10Y.json")

NICK_TO_ABBR = {
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BRK", "NewJersey": "BRK",
    "Hornets": "CHO", "Bulls": "CHI", "Cavaliers": "CLE", "Mavericks": "DAL",
    "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GSW", "Golden State": "GSW",
    "Rockets": "HOU", "Pacers": "IND", "Clippers": "LAC", "Lakers": "LAL",
    "Grizzlies": "MEM", "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN",
    "Pelicans": "NOP", "Knicks": "NYK", "Thunder": "OKC", "Magic": "ORL",
    "Seventysixers": "PHI", "Suns": "PHO", "Trailblazers": "POR", "Kings": "SAC",
    "Spurs": "SAS", "Raptors": "TOR", "Jazz": "UTA", "Wizards": "WAS",
}


def ml_to_prob(ml):
    """American moneyline -> implied probability (with vig)."""
    ml = float(ml)
    if ml < 0:
        return -ml / (-ml + 100.0)
    return 100.0 / (ml + 100.0)


def season_of(date_str):
    """NBA season end-year for a YYYY-MM-DD date (Aug rollover)."""
    y, m = int(date_str[:4]), int(date_str[5:7])
    return y + 1 if m >= 8 else y


def download():
    if not RAW.exists():
        print(f"downloading odds archive -> {RAW.name}")
        with urllib.request.urlopen(SRC, timeout=120) as r:
            RAW.write_bytes(r.read())
    return json.loads(RAW.read_text())


def main():
    data = download()
    rows = []
    for r in data:
        h, a = str(r.get("home_team")), str(r.get("away_team"))
        if h not in NICK_TO_ABBR or a not in NICK_TO_ABBR:
            continue
        hml, aml = r.get("home_close_ml"), r.get("away_close_ml")
        if not hml or not aml:
            continue
        try:
            d = str(int(float(r["date"])))
        except (TypeError, ValueError):
            continue
        date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        ph_raw, pa_raw = ml_to_prob(hml), ml_to_prob(aml)
        s = ph_raw + pa_raw
        rows.append({
            "DATE": date, "SEASON": season_of(date),
            "HOME": NICK_TO_ABBR[h], "AWAY": NICK_TO_ABBR[a],
            "ML_HOME": int(hml), "ML_AWAY": int(aml),
            "P_HOME": round(ph_raw / s, 4), "P_AWAY": round(pa_raw / s, 4),
        })
    df = pd.DataFrame(rows)
    cols = ["DATE", "HOME", "AWAY", "ML_HOME", "ML_AWAY", "P_HOME", "P_AWAY"]
    for season, g in df.groupby("SEASON"):
        if season < 2015 or season > 2026:
            continue
        out = CACHE / f"odds_{season}.csv"
        g.sort_values("DATE")[cols].to_csv(out, index=False)
        print(f"season {season}: {len(g)} games with moneylines -> {out.name}")
    print("note: archive covers ~2015-2022; run fetch_pinnacle_odds.py for 2023+.")


if __name__ == "__main__":
    main()
