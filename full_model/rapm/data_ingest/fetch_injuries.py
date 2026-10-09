"""Collect ESPN's NBA injury report as a timestamped CHANGE LOG (forward-only data).

There is no historical injury-report archive, and pre-tip availability is the largest
identified gap vs the closing line (market_gap.py: the log-loss gap is ~2.6x larger in
games where a top-2 rotation player sits). So we start recording now: every run diffs
the live report against the last known state and appends

    injury_log.csv   snapshot_utc, espn_ts, team, espn_team_id, athlete_id, name,
                     status, designation, report_date, short_comment, long_comment, event
                     event in {NEW, CHANGED, CLEARED}
    state.json       last known report per athlete (for the diff)

so the exact report as of any tip-off can be reconstructed (latest event <= tip time).
`designation` parses the comment for the official game status (out / doubtful /
questionable / probable / available / GTD), which ESPN's coarse `status` lacks.

Stdlib only (runs on a bare GitHub runner). Output dir: $INJURY_DIR or cache/injuries.
Run:  python data_ingest/fetch_injuries.py
"""
import csv
import datetime as dt
import json
import os
import re
import urllib.request
from pathlib import Path

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
UA = {"User-Agent": "Mozilla/5.0 (BOOKER injury log)"}
ABBR_FIX = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
            "BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}     # basketball-reference codes
OUT = Path(os.environ.get("INJURY_DIR", Path(__file__).resolve().parent.parent / "cache" / "injuries"))
FIELDS = ["snapshot_utc", "espn_ts", "team", "espn_team_id", "athlete_id", "name", "status",
          "designation", "report_date", "short_comment", "long_comment", "event"]
# ordered: first match wins (a "ruled out" comment that also says "questionable earlier" -> out)
_DESIG = [("out", r"\bruled out\b|\bwill (?:not|miss)\b|\bwon't play\b|\bout for\b|\bsidelined\b|\bwill sit\b|\bnot play\b"),
          ("doubtful", r"\bdoubtful\b"),
          ("gtd", r"\bgame-time decision\b"),
          ("questionable", r"\bquestionable\b"),
          ("probable", r"\bprobable\b"),
          ("available", r"\bwill play\b|\bcleared to play\b|\bavailable\b|\bupgraded to available\b|\bin the starting lineup\b")]


def _get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def designation(status, short, long_):
    text = f"{short} {long_}".lower()
    for lab, pat in _DESIG:
        if re.search(pat, text):
            return lab
    s = (status or "").lower()
    return "out" if s == "out" else ("dtd" if "day" in s else s or "unknown")


def team_abbrs():
    js = _get(TEAMS_URL)
    teams = [t["team"] for t in js["sports"][0]["leagues"][0]["teams"]]
    return {str(t["id"]): ABBR_FIX.get(t["abbreviation"], t["abbreviation"]) for t in teams}


def snapshot():
    js = _get(URL)
    abbr = team_abbrs()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = {}
    for t in js.get("injuries", []):
        tid = str(t.get("id"))
        for i in t.get("injuries", []):
            a = i.get("athlete", {})
            href = next((l.get("href", "") for l in a.get("links", []) if "/id/" in l.get("href", "")), "")
            m = re.search(r"/id/(\d+)", href)
            aid = m.group(1) if m else a.get("displayName", "")
            short, long_ = i.get("shortComment", "") or "", i.get("longComment", "") or ""
            rows[aid] = {
                "snapshot_utc": now, "espn_ts": js.get("timestamp", ""), "team": abbr.get(tid, tid),
                "espn_team_id": tid, "athlete_id": aid, "name": a.get("displayName", ""),
                "status": i.get("status", ""), "designation": designation(i.get("status"), short, long_),
                "report_date": i.get("date", ""), "short_comment": short.replace("\n", " "),
                "long_comment": long_.replace("\n", " ")[:600]}
    return now, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state_p, log_p = OUT / "state.json", OUT / "injury_log.csv"
    prev = json.loads(state_p.read_text()) if state_p.exists() else {}
    now, cur = snapshot()
    events = []
    key = lambda r: (r["status"], r["designation"], r["report_date"], r["short_comment"], r["team"])
    for aid, r in cur.items():
        if aid not in prev:
            events.append({**r, "event": "NEW"})
        elif key(prev[aid]) != key(r):
            events.append({**r, "event": "CHANGED"})
    for aid, r in prev.items():
        if aid not in cur:
            events.append({**r, "snapshot_utc": now, "event": "CLEARED"})
    new_file = not log_p.exists()
    with log_p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(events)
    state_p.write_text(json.dumps(cur, indent=0, sort_keys=True))
    print(f"{now}: {len(cur)} listed, {len(events)} events "
          f"({sum(e['event']=='NEW' for e in events)} new, {sum(e['event']=='CHANGED' for e in events)} changed, "
          f"{sum(e['event']=='CLEARED' for e in events)} cleared) -> {log_p}")


if __name__ == "__main__":
    main()
