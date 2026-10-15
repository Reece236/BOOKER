"""
Historical NBA closing (and opening) moneylines from ESPN's core odds API.

For each regular-season event ESPN lists ~5-12 sportsbooks with open and close American
moneylines. Per book we remove the vig (proportional), then take the MEDIAN across books
as the consensus price (robust to a stale book). Opening consensus is kept for CLV work.

DATA TRAPS (verified): ESPN also lists "... - Live Odds" feeds whose stored "close" is an
IN-GAME / post-game price (2023-24 and 2024-25 pregame moneylines are otherwise absent --
using them gave 94% accuracy, LL .17). Those feeds, and model/tout providers (accuscore,
teamrankings, numberfire, betegy, consensus), are excluded. When a game has no pregame
moneyline, P_HOME comes from the median pregame SPREAD via P = Phi(-(spread - B)/SIGMA),
fit on 2015-22 (n=9,612) and validated on 2022-23: spread-implied LL .6235 vs moneyline
consensus .6231. SOURCE = "espn" (moneyline) or "espn_spread".

Raw per-book rows   -> cache/odds_espn_books_{season}.csv   (private; never published)
Consensus per game  -> cache/odds_espn_{season}.csv          schema = odds_{season}.csv
                       + P_HOME_OPEN, N_BOOKS
`--merge` fills games missing from cache/odds_{season}.csv (existing sources win).

Usage:  python -m data_ingest.fetch_espn_odds 2024 2025 [--merge]
Season = end year (2024 = 2023-24).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
ABBR_FIX = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
            "BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}


def _get(url, tries=4):
    for k in range(tries):
        r = subprocess.run(["curl", "-s", "--compressed", "-m", "30", url], capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def events(season):
    out = []
    for y, m in [(season - 1, mm) for mm in (10, 11, 12)] + [(season, mm) for mm in (1, 2, 3, 4)]:
        d = _get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={y}{m:02d}&limit=500")
        for e in (d or {}).get("events", []):
            if e.get("season", {}).get("type") != 2:
                continue
            c = e["competitions"][0]
            tm = {t["homeAway"]: ABBR_FIX.get(t["team"]["abbreviation"], t["team"]["abbreviation"]) for t in c["competitors"]}
            dt = pd.Timestamp(e["date"]).tz_convert("US/Eastern").strftime("%Y-%m-%d")
            out.append({"eid": e["id"], "DATE": dt, "HOME": tm["home"], "AWAY": tm["away"]})
    return pd.DataFrame(out).drop_duplicates("eid")


def _am(o):
    if o is None:
        return np.nan
    v = o.get("american") if isinstance(o, dict) else o
    try:
        return float(str(v).replace("+", "")) if v not in (None, "", "EVEN") else (100.0 if v == "EVEN" else np.nan)
    except ValueError:
        return np.nan


def odds_for(eid):
    d = _get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds?limit=50")
    rows = []
    for it in (d or {}).get("items", []):
        h, a = it.get("homeTeamOdds", {}) or {}, it.get("awayTeamOdds", {}) or {}
        rows.append({"eid": eid, "book": (it.get("provider") or {}).get("name"),
                     "ml_home_close": _am((h.get("close") or {})) if h.get("close") else _am(h.get("moneyLine")),
                     "ml_away_close": _am((a.get("close") or {})) if a.get("close") else _am(a.get("moneyLine")),
                     "ml_home_open": _am(h.get("open")), "ml_away_open": _am(a.get("open")),
                     "spread": it.get("spread"), "total": it.get("overUnder")})
    return rows


def imp(ml):
    ml = np.asarray(ml, float)
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


SPREAD_B, SPREAD_SIGMA = 0.02, 11.72          # fit 2015-22, see docstring
NOT_BOOKS = {"consensus", "teamrankings", "accuscore", "numberfire", "betegy"}


def consensus(books):
    from scipy.stats import norm
    b = books[~books.book.fillna("").str.contains("Live") & ~books.book.fillna("").str.lower().isin(NOT_BOOKS)].copy()
    b["spread"] = pd.to_numeric(b.spread, errors="coerce").where(lambda s: s.abs() < 30)
    for k in ("close", "open"):
        ph, pa = imp(b[f"ml_home_{k}"]), imp(b[f"ml_away_{k}"])
        ok = np.isfinite(ph) & np.isfinite(pa) & (ph + pa > 0.98) & (ph + pa < 1.12)   # sane two-way books only
        b[f"p_{k}"] = np.where(ok, ph / (ph + pa), np.nan)
    g = b.groupby("eid")
    out = pd.DataFrame({"P_HOME": g.p_close.median(), "P_HOME_OPEN": g.p_open.median(),
                        "N_BOOKS": g.p_close.count(), "HOME_SPREAD": g.spread.median(),
                        "OVER_TOTAL": g.total.median(),
                        "ML_HOME": g.ml_home_close.median(), "ML_AWAY": g.ml_away_close.median()}).reset_index()
    out["SOURCE"] = np.where(out.P_HOME.notna(), "espn", "espn_spread")
    sp = norm.cdf(-(out.HOME_SPREAD - SPREAD_B) / SPREAD_SIGMA)
    out["P_HOME"] = out.P_HOME.fillna(pd.Series(sp, index=out.index))
    out["N_BOOKS"] = np.where(out.N_BOOKS > 0, out.N_BOOKS, g.spread.count().values)
    out["P_AWAY"] = 1 - out.P_HOME
    return out[out.P_HOME.notna()]


def run(season, merge=False):
    ev = events(season)
    print(f"{season}: {len(ev)} regular-season events", flush=True)
    with ThreadPoolExecutor(8) as ex:
        rows = [r for rs in ex.map(odds_for, ev.eid) for r in rs]
    books = pd.DataFrame(rows).merge(ev, on="eid")
    books.to_csv(CACHE / f"odds_espn_books_{season}.csv", index=False)
    c = consensus(books).merge(ev, on="eid")
    cols = ["DATE", "HOME", "AWAY", "ML_HOME", "ML_AWAY", "P_HOME", "P_AWAY", "HOME_SPREAD", "OVER_TOTAL",
            "SOURCE", "P_HOME_OPEN", "N_BOOKS"]
    c = c[cols].sort_values("DATE")
    c.to_csv(CACHE / f"odds_espn_{season}.csv", index=False)
    print(f"  consensus lines for {len(c)} games (median {c.N_BOOKS.median():.0f} books)", flush=True)
    if merge:
        p = CACHE / f"odds_{season}.csv"
        if p.exists():
            old = pd.read_csv(p)
            key = old.DATE.astype(str) + old.HOME + old.AWAY
            add = c[~(c.DATE + c.HOME + c.AWAY).isin(key)]
            new = pd.concat([old, add[old.columns.intersection(add.columns)]], ignore_index=True)
        else:
            add, new = c, c.drop(columns=["P_HOME_OPEN", "N_BOOKS"])
        new.sort_values("DATE").to_csv(p, index=False)
        print(f"  merged into {p.name}: +{len(add)} games -> {len(new)}", flush=True)
    return c


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for s in args:
        run(int(s), merge="--merge" in sys.argv)
