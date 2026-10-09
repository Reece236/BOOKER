"""Pre-tip availability from the collected ESPN injury log (data-injuries branch).

report_as_of(ts)        each athlete's latest report with snapshot_utc <= ts
p_play(designation)     P(plays) for a designation -- PRIORS until calibrated
calibrate()             fit P(plays | designation) from the log joined to who played
                        (run once a season of logs exists); writes cache/injury_calibration.json
availability_override(team, game_ts, name_to_pid)   {pid: p_play} for forecast.sequential

The log lives at $INJURY_DIR/injury_log.csv if set, else cache/injuries/injury_log.csv, else
it is fetched from the data-injuries branch on GitHub.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

from . import player_impacts as pi

CACHE = pi.CACHE
RAW = "https://raw.githubusercontent.com/Reece236/BOOKER/data-injuries/injuries/injury_log.csv"
# Priors (NBA official-designation conventions; replace via calibrate()).
PRIORS = {"out": 0.02, "doubtful": 0.15, "questionable": 0.50, "gtd": 0.55, "dtd": 0.60,
          "probable": 0.90, "available": 0.98}
CAL = CACHE / "injury_calibration.json"


def load_log():
    for p in (Path(os.environ.get("INJURY_DIR", "")) / "injury_log.csv", CACHE / "injuries" / "injury_log.csv"):
        if str(p) != "injury_log.csv" and p.exists():
            return pd.read_csv(p)
    with urllib.request.urlopen(RAW, timeout=60) as r:
        return pd.read_csv(io.StringIO(r.read().decode()))


def report_as_of(log, ts):
    """Latest event per athlete at or before UTC timestamp `ts` (ISO string); CLEARED drops."""
    L = log[log.snapshot_utc <= ts].sort_values("snapshot_utc").groupby("athlete_id").tail(1)
    return L[L.event != "CLEARED"]


def p_play(designation):
    table = dict(PRIORS)
    if CAL.exists():
        table.update(json.loads(CAL.read_text()))
    return table.get(str(designation).lower(), 0.6)


def availability_override(log, team, game_ts, name_to_pid):
    rep = report_as_of(log, game_ts)
    rep = rep[rep.team == team]
    out = {}
    for nm, des in zip(rep.name, rep.designation):
        pid = name_to_pid.get(pi.norm_name(nm))
        if pid is not None:
            out[pid] = p_play(des)
    return out


def calibrate(played_by_game):
    """played_by_game: DataFrame [date, team, name, played] (who actually suited up).
    Joins each team-game's report as of 22:00 UTC that day (pre-tip for most games)."""
    log = load_log()
    rows = []
    for (date, team), g in played_by_game.groupby(["date", "team"]):
        rep = report_as_of(log, f"{date}T22:00:00Z")
        rep = rep[rep.team == team]
        pl = dict(zip(g.name.map(pi.norm_name), g.played))
        for nm, des in zip(rep.name, rep.designation):
            k = pi.norm_name(nm)
            if k in pl:
                rows.append((des, pl[k]))
    if not rows:
        return {}
    df = pd.DataFrame(rows, columns=["designation", "played"])
    cal = df.groupby("designation").played.agg(["mean", "size"])
    cal = cal[cal["size"] >= 30]["mean"].round(3).to_dict()
    CAL.write_text(json.dumps(cal, indent=1))
    return cal
