"""Build season rows for full_model/nba_player_data_2015-2025.csv from basketball-reference.

The file's schema (bbref ids, `2TM` multi-team codes, `box` = BPM, OWS/DWS/WS/VORP)
comes from bbref's season totals + advanced tables. This fetcher joins the two
tables on (player id, TEAM) -- the original builder joined on player only, so every
traded player's rows were a cartesian product (De'Aaron Fox 2025: 3 team rows x 3
minutes values = 9 rows, each team paired with the wrong minutes/advanced stats).

Usage:
    python data_ingest/fetch_bbref_season.py 2026                 # append/replace 2026
    python data_ingest/fetch_bbref_season.py --validate 2025      # compare vs file, no write
    python data_ingest/fetch_bbref_season.py --repair 2015 ... 2025   # rebuild seasons
Polite: one page every ~4 s (bbref allows ~20 requests/min), cached in /tmp.
"""
import html
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FILE = HERE.parent.parent / "nba_player_data_2015-2025.csv"
TMP = Path("/tmp/bbref_cache")
TMP.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (BOOKER research; low-rate)"}

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r'<(?:td|th)([^>]*)data-stat="([a-z_0-9]+)"([^>]*)>(.*?)</(?:td|th)>', re.S)
_ID = re.compile(r'data-append-csv="([^"]+)"')
_TAG = re.compile(r"<[^>]+>")


def _page(season, kind):
    f = TMP / f"NBA_{season}_{kind}.html"
    if not f.exists():
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_{kind}.html"
        txt = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8")
        f.write_text(txt)
        time.sleep(4.0)
    return f.read_text()


def _table(season, kind):
    txt = _page(season, kind)
    rows = []
    for tr in _ROW.findall(txt):
        cells = _CELL.findall(tr)
        if not cells:
            continue
        rec = {}
        for pre, stat, post, val in cells:
            if stat == "name_display":
                m = _ID.search(pre + post)
                rec["playerId"] = m.group(1) if m else None
            rec[stat] = html.unescape(_TAG.sub("", val)).strip()
        if rec.get("playerId") and rec.get("team_name_abbr"):
            rows.append(rec)
    df = pd.DataFrame(rows).drop_duplicates(["playerId", "team_name_abbr"])
    return df


def _num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def build_season(season):
    tot = _table(season, "totals")
    adv = _table(season, "advanced")
    m = tot.merge(adv[["playerId", "team_name_abbr", "ts_pct", "usg_pct", "ows", "dws", "ws",
                       "vorp", "bpm"]], on=["playerId", "team_name_abbr"], how="left")
    out = pd.DataFrame({
        "playerId": m.playerId, "playerName": m.name_display, "season": season,
        "age": _num(m.age), "team": m.team_name_abbr, "position": m.pos,
        "games": _num(m.games), "minutesPlayed": _num(m.mp),
        "total_points": _num(m.pts), "total_assists": _num(m.ast), "total_totalRb": _num(m.trb),
        "total_offensiveRb": _num(m.orb), "total_defensiveRb": _num(m.drb),
        "total_steals": _num(m.stl), "total_blocks": _num(m.blk), "total_turnovers": _num(m.tov),
        "total_personalFouls": _num(m.pf), "total_fieldAttempts": _num(m.fga),
        "total_fieldGoals": _num(m.fg), "total_threeAttempts": _num(m.fg3a),
        "total_threeFg": _num(m.fg3), "total_ftAttempts": _num(m.fta), "total_ft": _num(m.ft),
        "tsPercent": _num(m.ts_pct), "usagePercent": _num(m.usg_pct),
        "offensiveWS": _num(m.ows), "defensiveWS": _num(m.dws), "winShares": _num(m.ws),
        "vorp": _num(m.vorp), "box": _num(m.bpm),
    })
    return out


def validate(season):
    old = pd.read_csv(FILE)
    old = old[old.season == season]
    new = build_season(season)
    single = old.groupby("playerId").filter(lambda g: len(g) == 1)
    j = single.merge(new, on=["playerId", "team"], suffixes=("_old", ""))
    print(f"{season}: file {len(old)} rows / bbref {len(new)} rows; single-team players matched {len(j)}/{len(single)}")
    for c in ("minutesPlayed", "total_points", "usagePercent", "offensiveWS", "vorp", "box", "tsPercent"):
        d = (j[c] - j[c + "_old"]).abs()
        print(f"  {c:15s} exact {np.mean(d < 1e-6):.3f}  max|diff| {d.max():.3f}")
    multi = old.groupby("playerId").filter(lambda g: len(g) > 1)
    print(f"  traded players: file {multi.playerId.nunique()} players in {len(multi)} rows "
          f"(bbref has {new[new.playerId.isin(multi.playerId)].shape[0]} rows for them)")
    return new


def write(seasons):
    df = pd.read_csv(FILE)
    for s in seasons:
        new = build_season(s)
        df = pd.concat([df[df.season != s], new], ignore_index=True)
        print(f"  {s}: {len(new)} rows ({new.playerId.nunique()} players)")
    df = df.sort_values(["playerName", "season"], kind="stable")
    df.to_csv(FILE, index=False)
    print(f"wrote {FILE} ({len(df)} rows, seasons {df.season.min()}-{df.season.max()})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--validate" in a:
        for s in [int(x) for x in a if x.isdigit()]:
            validate(s)
    else:
        write([int(x) for x in a if x.isdigit()])
