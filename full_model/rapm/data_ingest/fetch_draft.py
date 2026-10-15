"""Draft history + team executive tenures from basketball-reference.

    cache/draft_history.csv   draft_year, pick, team (our codes), bbref_id, player, college
    cache/team_execs.csv      team, exec, start, end   (top basketball executive tenures)

The drafting team is the team that MADE the pick (bbref's Tm column); draft-night trades
are not followed. The exec credited for a draft is the one in office on draft night.
Polite: one page every ~4 s, cached in /tmp/bbref_cache.
Run:  python data_ingest/fetch_draft.py [first_year] [last_year]
"""
import html
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
TMP = Path("/tmp/bbref_cache"); TMP.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (BOOKER research; low-rate)"}
# draft-page codes -> our codes (franchise renames)
TEAM_FIX = {"NJN": "BRK", "CHA": "CHO", "NOH": "NOP", "NOK": "NOP", "SEA": "OKC", "VAN": "MEM"}
# our codes -> bbref FRANCHISE code for the executives page
FRANCHISE = {"BRK": "NJN", "CHO": "CHA", "NOP": "NOH"}
TEAMS = ["ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC",
         "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHO", "POR", "SAC",
         "SAS", "TOR", "UTA", "WAS"]
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r'<(?:td|th)([^>]*)data-stat="([a-z_0-9]+)"([^>]*)>(.*?)</(?:td|th)>', re.S)


def _page(url, key):
    f = TMP / key
    if not f.exists():
        f.write_text(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8"))
        time.sleep(4.0)
    return f.read_text()


def _rows(txt):
    for tr in _ROW.findall(txt):
        rec = {}
        for pre, stat, post, val in _CELL.findall(tr):
            m = re.search(r'data-append-csv="([^"]+)"', pre + post)
            if m:
                rec[stat + "_id"] = m.group(1)
            rec[stat] = html.unescape(re.sub(r"<[^>]+>", "", val)).strip()
        if rec:
            yield rec


def drafts(first, last):
    out = []
    for y in range(first, last + 1):
        txt = _page(f"https://www.basketball-reference.com/draft/NBA_{y}.html", f"draft_{y}.html")
        for r in _rows(txt):
            if not r.get("pick_overall", "").isdigit() or not r.get("player"):
                continue
            tm = r.get("team_id", "")
            out.append({"draft_year": y, "pick": int(r["pick_overall"]), "team": TEAM_FIX.get(tm, tm),
                        "bbref_id": r.get("player_id"), "player": r["player"],
                        "college": r.get("college_name", "")})
    return pd.DataFrame(out)


def execs():
    out = []
    for t in TEAMS:
        code = FRANCHISE.get(t, t)
        try:
            txt = _page(f"https://www.basketball-reference.com/teams/{code}/executives.html", f"exec_{code}.html")
        except Exception as exc:                           # noqa: BLE001
            print(f"  {t}: executives page failed ({exc})"); continue
        for r in _rows(txt):
            if r.get("exec") and r.get("date_start") and r["exec"] != "Executive":
                out.append({"team": t, "exec": r["exec"], "start": r["date_start"],
                            "end": r.get("date_end") or "2099-12-31"})
    return pd.DataFrame(out)


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:] if x.isdigit()]
    first, last = (a + [2012, 2026])[:2] if len(a) < 2 else a[:2]
    d = drafts(first, last)
    d.to_csv(CACHE / "draft_history.csv", index=False)
    print(f"draft_history.csv: {len(d)} picks, {d.draft_year.min()}-{d.draft_year.max()}")
    e = execs()
    e.to_csv(CACHE / "team_execs.csv", index=False)
    print(f"team_execs.csv: {len(e)} tenures, {e.team.nunique()} teams")
