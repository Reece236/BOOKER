"""
Unified NBA betting-line ingest: Sportsbook Review moneylines + VegasLines (Covers) spreads.

Sources
-------
1. **SBR archive** (flancast90/sportsbookreview-scraper) — closing moneylines, ~2015-2022.
2. **VegasLines / Covers** (papagorgio23/VegasLines) — historical home spread + O/U.
   The bundled VegasLines2000-2017.csv in that repo is **NFL only**. NBA requires the
   GetLines() R scraper (Covers.com), which is currently broken in most environments.
   This script can invoke GetLines via R when available and converts spreads to
   vig-removed win probabilities.

Writes cache/odds_{season}.csv:
    DATE, HOME, AWAY, ML_HOME, ML_AWAY, P_HOME, P_AWAY, HOME_SPREAD, SOURCE

For seasons with spread but no ML, P_HOME/P_AWAY come from the spread-implied model.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RAW_SBR = CACHE / "odds_archive.json"
RAW_VEGAS = CACHE / "vegaslines_nba_raw.csv"
VEGAS_REPO = "https://github.com/papagorgio23/VegasLines.git"
SBR_SRC = (
    "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/"
    "main/data/nba_archive_10Y.json"
)

# Covers / VegasLines team nicknames -> BOOKER abbr
VEGAS_TO_ABBR = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BRK", "Charlotte": "CHO",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LA Clippers": "LAC", "L.A. Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "LA Lakers": "LAL", "L.A. Lakers": "LAL", "Los Angeles Lakers": "LAL",
    "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "New Orleans": "NOP", "New York": "NYK", "Oklahoma City": "OKC",
    "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHO", "Portland": "POR",
    "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR", "Utah": "UTA",
    "Washington": "WAS",
}

SBR_NICK = {
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BRK", "NewJersey": "BRK",
    "Hornets": "CHO", "Bulls": "CHI", "Cavaliers": "CLE", "Mavericks": "DAL",
    "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GSW", "Golden State": "GSW",
    "Rockets": "HOU", "Pacers": "IND", "Clippers": "LAC", "Lakers": "LAL",
    "Grizzlies": "MEM", "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN",
    "Pelicans": "NOP", "Knicks": "NYK", "Thunder": "OKC", "Magic": "ORL",
    "Seventysixers": "PHI", "Suns": "PHO", "Trailblazers": "POR", "Kings": "SAC",
    "Spurs": "SAS", "Raptors": "TOR", "Jazz": "UTA", "Wizards": "WAS",
}


def season_of(date_str):
    y, m = int(date_str[:4]), int(date_str[5:7])
    return y + 1 if m >= 8 else y


def ml_to_prob(ml):
    ml = float(ml)
    if ml < 0:
        return -ml / (-ml + 100.0)
    return 100.0 / (ml + 100.0)


def spread_to_win_prob(home_line, sigma=11.5):
    """
    Covers home.line -> P(home win). Negative line = home favored by |line| points.
    """
    if home_line is None or (isinstance(home_line, float) and np.isnan(home_line)):
        return np.nan
    margin = -float(home_line)
    return float(norm.cdf(margin / sigma))


def download_sbr():
    if not RAW_SBR.exists():
        print(f"downloading SBR archive -> {RAW_SBR.name}")
        with urllib.request.urlopen(SBR_SRC, timeout=120) as r:
            RAW_SBR.write_bytes(r.read())
    return json.loads(RAW_SBR.read_text())


def load_sbr_rows():
    data = download_sbr()
    rows = []
    for r in data:
        h, a = str(r.get("home_team")), str(r.get("away_team"))
        if h not in SBR_NICK or a not in SBR_NICK:
            continue
        hml, aml = r.get("home_close_ml"), r.get("away_close_ml")
        if not hml or not aml:
            continue
        try:
            d = str(int(float(r["date"])))
        except (TypeError, ValueError):
            continue
        date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        ph, pa = ml_to_prob(hml), ml_to_prob(aml)
        s = ph + pa
        rows.append({
            "DATE": date, "HOME": SBR_NICK[h], "AWAY": SBR_NICK[a],
            "ML_HOME": int(hml), "ML_AWAY": int(aml),
            "P_HOME": round(ph / s, 4), "P_AWAY": round(pa / s, 4),
            "HOME_SPREAD": np.nan, "SOURCE": "sbr_ml",
        })
    return pd.DataFrame(rows)


def run_getlines_r(year_start, year_end):
    """Try VegasLines GetLinesRange via R (Covers.com). Returns empty df on failure."""
    import tempfile
    r_script = f"""
    suppressPackageStartupMessages({{
      library(lubridate); library(dplyr); library(rvest); library(magrittr)
    }})
    lines <- readLines("GetLines.R")
    eval(parse(text=paste(lines[1:141], collapse="\\n")))
    d <- GetLinesRange("NBA", {year_start}, {year_end}, "regular season")
    write.csv(d, "{RAW_VEGAS}", row.names=FALSE)
    """
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "VegasLines"
        subprocess.run(["git", "clone", "--depth", "1", VEGAS_REPO, str(repo)],
                       capture_output=True, check=True, timeout=120)
        out = subprocess.run(
            ["Rscript", "-e", r_script.replace("GetLines.R", str(repo / "GetLines.R"))
             .replace(str(RAW_VEGAS), str(repo / "out.csv"))],
            cwd=str(repo), capture_output=True, text=True, timeout=900, check=False,
        )
        out_csv = repo / "out.csv"
        if out.returncode != 0 or not out_csv.exists():
            print("VegasLines scrape failed:", (out.stderr or "")[-300:])
            return pd.DataFrame()
        return pd.read_csv(out_csv)


def vegas_df_to_rows(df):
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for r in df.itertuples():
        ht = VEGAS_TO_ABBR.get(str(r.home_team).strip())
        at = VEGAS_TO_ABBR.get(str(r.away_team).strip())
        if not ht or not at:
            continue
        date = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        spread = getattr(r, "home_line", np.nan)
        ph = spread_to_win_prob(spread)
        if np.isnan(ph):
            continue
        rows.append({
            "DATE": date, "HOME": ht, "AWAY": at,
            "ML_HOME": np.nan, "ML_AWAY": np.nan,
            "P_HOME": round(ph, 4), "P_AWAY": round(1 - ph, 4),
            "HOME_SPREAD": spread, "SOURCE": "vegaslines_spread",
        })
    return pd.DataFrame(rows)


def merge_odds(sbr, vegas):
    """Prefer SBR moneylines; fill gaps with VegasLines spread-implied probs."""
    if sbr.empty and vegas.empty:
        return sbr
    key = ["DATE", "HOME", "AWAY"]
    if sbr.empty:
        return vegas
    if vegas.empty:
        return sbr
    m = sbr.merge(vegas[key + ["P_HOME", "HOME_SPREAD", "SOURCE"]],
                  on=key, how="outer", suffixes=("", "_vegas"))
    for c in ("P_HOME", "ML_HOME", "ML_AWAY", "HOME_SPREAD", "SOURCE"):
        if c not in m.columns:
            m[c] = np.nan
    use_vegas = m.P_HOME.isna() & m.P_HOME_vegas.notna()
    m.loc[use_vegas, "P_HOME"] = m.loc[use_vegas, "P_HOME_vegas"]
    m.loc[use_vegas, "P_AWAY"] = 1 - m.loc[use_vegas, "P_HOME"]
    m.loc[use_vegas, "SOURCE"] = m.loc[use_vegas, "SOURCE_vegas"]
    m.loc[m.HOME_SPREAD.isna(), "HOME_SPREAD"] = m.loc[m.HOME_SPREAD.isna(), "HOME_SPREAD_vegas"]
    drop = [c for c in m.columns if c.endswith("_vegas")]
    m = m.drop(columns=drop)
    return m


def write_season_files(df, max_season=2026):
    cols = ["DATE", "HOME", "AWAY", "ML_HOME", "ML_AWAY", "P_HOME", "P_AWAY",
            "HOME_SPREAD", "SOURCE"]
    df["SEASON"] = df.DATE.map(season_of)
    for season, g in df.groupby("SEASON"):
        if season < 2015 or season > max_season:
            continue
        out = CACHE / f"odds_{season}.csv"
        g.sort_values("DATE")[cols].to_csv(out, index=False)
        src = g.SOURCE.value_counts().to_dict()
        print(f"season {season}: {len(g)} games -> {out.name}  sources={src}")


def main():
    print("=== SBR closing moneylines ===")
    sbr = load_sbr_rows()

    print("=== VegasLines (Covers) via R GetLines 2015-2026 ===")
    vegas = pd.DataFrame()
    if RAW_VEGAS.exists():
        print(f"loading cached {RAW_VEGAS.name}")
        vegas = vegas_df_to_rows(pd.read_csv(RAW_VEGAS))
    else:
        scraped = run_getlines_r(2015, 2026)
        if not scraped.empty:
            scraped.to_csv(RAW_VEGAS, index=False)
            vegas = vegas_df_to_rows(scraped)
        else:
            print("VegasLines live scrape unavailable; using SBR only for ML seasons.")
            print("To add Covers data: run GetLinesRange in R and save to", RAW_VEGAS)

    merged = merge_odds(sbr, vegas)
    write_season_files(merged, max_season=2026)

    print("=== Local odds CSVs (oddsData + Pinnacle) ===")
    try:
        from fetch_local_odds import main as local_odds_main
        local_odds_main()
    except Exception as exc:
        print(f"local odds ingest skipped: {exc}")

    print("done.")


if __name__ == "__main__":
    main()
