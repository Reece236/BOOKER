"""Fetch the REAL regular-season schedule for a season from ESPN -> cache/games_{season}.csv.

Replaces the cloned placeholder (clone_schedule_2027.py). ESPN's scoreboard returns a
whole month per request (dates=YYYYMM); we keep regular-season events (season.type 2),
convert tip times to US/Eastern calendar dates, and map ESPN abbreviations to our
basketball-reference codes.

NBA Cup: each team has 80 fixed games; its last 2 are set after the Cup group stage
(ESPN shows 'TBD' knockout placeholders until then). TBD rows are dropped and the 30
missing games are added as PLACEHOLDER=1 rows (a balanced cycle, each team exactly 2
games, mid-December) so the season sim still plays 82. Re-run after the league
publishes them to get the real matchups.

GAME_ID: the pipeline treats str(GAME_ID)[0] == '2' as regular season and '4' as playoffs,
and ESPN event ids start with 4 -- so we assign NBA-convention ids 2YY00NNNN in tip-off
order and keep ESPN's id in ESPN_ID (for joining results/box scores later).

Run:  python data_ingest/fetch_schedule.py 2027 [--write]
"""
import datetime as dt
import gzip
import json
import sys
import time
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ym}&limit=1000"
ABBR_FIX = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
            "BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}
ET = ZoneInfo("America/New_York")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return json.loads(raw)


def fetch(season):
    months = [(season - 1, m) for m in (10, 11, 12)] + [(season, m) for m in (1, 2, 3, 4)]
    rows = {}
    for y, m in months:
        d = _get(URL.format(ym=f"{y}{m:02d}"))
        for e in d.get("events", []):
            if e.get("season", {}).get("type") != 2 or e.get("season", {}).get("year") != season:
                continue
            c = e["competitions"][0]
            side = {x["homeAway"]: x for x in c["competitors"]}
            ab = lambda x: ABBR_FIX.get(x["team"]["abbreviation"], x["team"]["abbreviation"])
            tip = dt.datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
            done = e["status"]["type"].get("completed", False)
            hp = int(side["home"]["score"]) if done else None
            ap = int(side["away"]["score"]) if done else None
            rows[e["id"]] = {"ESPN_ID": int(e["id"]), "TIP_UTC": tip.strftime("%Y-%m-%dT%H:%MZ"),
                             "DATE": tip.astimezone(ET).strftime("%Y-%m-%d"), "SEASON": season,
                             "HOME": ab(side["home"]), "AWAY": ab(side["away"]),
                             "HOME_PTS": hp, "AWAY_PTS": ap, "SEASON_TYPE": "Regular Season",
                             "HOME_WIN": (None if hp is None else int(hp > ap))}
        time.sleep(0.5)
    g = pd.DataFrame(rows.values())
    g = g[(g.HOME != "TBD") & (g.AWAY != "TBD")]
    g["PLACEHOLDER"] = 0
    per = pd.concat([g.HOME, g.AWAY]).value_counts()
    short = sorted(per[per < 82].index)
    need = {t: 82 - int(per[t]) for t in short}
    if short and all(v == 2 for v in need.values()) and len(short) % 2 == 0:
        ph = []
        for i, tm in enumerate(short):                    # cycle: each team plays 2
            opp = short[(i + 1) % len(short)]
            day = f"{season - 1}-12-{15 + i % 6:02d}"
            ph.append({"ESPN_ID": None, "TIP_UTC": f"{day}T00:00Z", "DATE": day, "SEASON": season,
                       "HOME": tm, "AWAY": opp, "HOME_PTS": None, "AWAY_PTS": None,
                       "SEASON_TYPE": "Regular Season", "HOME_WIN": None, "PLACEHOLDER": 1})
        g = pd.concat([g, pd.DataFrame(ph)], ignore_index=True)
    g = g.sort_values(["TIP_UTC", "ESPN_ID"]).reset_index(drop=True)
    g.insert(0, "GAME_ID", [int(f"2{str(season - 1)[-2:]}00{i + 1:04d}") for i in range(len(g))])
    return g


def validate(g):
    teams = pd.concat([g.HOME, g.AWAY])
    per = teams.value_counts()
    home = g.HOME.value_counts()
    print(f"{len(g)} games, {per.size} teams | games/team {per.min()}-{per.max()} | "
          f"home games/team {home.min()}-{home.max()} | {g.DATE.min()} .. {g.DATE.max()}")
    print(f"  placeholders (post-Cup games not yet scheduled): {int(g.PLACEHOLDER.sum())}")
    return per.size == 30 and per.min() == 82 and per.max() == 82


if __name__ == "__main__":
    season = int(next((a for a in sys.argv[1:] if a.isdigit()), 2027))
    g = fetch(season)
    ok = validate(g)
    if "--write" in sys.argv:
        if not ok:
            sys.exit("schedule looks incomplete -- not written")
        out = CACHE / f"games_{season}.csv"
        if out.exists():
            out.rename(out.with_suffix(".csv.cloned_bak"))
        g.to_csv(out, index=False)
        print(f"wrote {out}")
