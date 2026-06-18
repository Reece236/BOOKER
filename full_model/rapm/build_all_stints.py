"""
Download + reconstruct lineup stints for every season we model and cache compact
per-season artifacts used by the out-of-sample backtest.

shufinskiy nbastats year = (our season) - 1.  Our seasons 2015-2025 -> shuf 2014-2024.

For each season we write to full_model/rapm/cache/:
  stints_{season}.csv      GAME_ID, PERIOD, HOME_LINEUP, AWAY_LINEUP, POSS, Y, DURATION_SECONDS, PLUS_MINUS
  players_{season}.csv     PLAYER_ID, NAME, TEAM_ID, MINUTES
  teams_{season}.csv       TEAM_ID, ABBR, ACTUAL_NET

The large raw nbastats csv is deleted after processing to save disk.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import nba_on_court as noc

from build_season_stints import build_lineup_stints, _sec, TEAM_ID_FLOOR

HERE = Path(__file__).resolve().parent
PBP = HERE / "pbp"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
PBP.mkdir(exist_ok=True)

SEASONS = list(range(2015, 2026))     # our seasons
SEC_PER_POSS = 28.8

TEAM_ID_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
    1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
    1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612751: "BRK", 1610612752: "NYK",
    1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHO",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
    1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
    1610612765: "DET", 1610612766: "CHO",
}


def parse(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def process_season(season):
    shuf = season - 1
    csv = PBP / f"nbastats_{shuf}.csv"
    if not csv.exists():
        noc.load_nba_data(path=str(PBP), seasons=shuf, data="nbastats", untar=True)
    df = pd.read_csv(csv, low_memory=False)
    df["GAME_ID"] = df["GAME_ID"].astype("int64")

    stints = build_lineup_stints(df)
    stints = stints[(stints.HOME_LINEUP.str.count(",") == 4)
                    & (stints.AWAY_LINEUP.str.count(",") == 4)].copy()
    stints["POSS"] = stints.DURATION_SECONDS / SEC_PER_POSS
    stints["Y"] = stints.PLUS_MINUS / stints.POSS * 100.0

    # ---- player names ---------------------------------------------------------
    names = {}
    for pid, nm in [("PLAYER1_ID", "PLAYER1_NAME"), ("PLAYER2_ID", "PLAYER2_NAME"),
                    ("PLAYER3_ID", "PLAYER3_NAME")]:
        t = df[[pid, nm]].dropna()
        for a, b in zip(t[pid].astype("int64"), t[nm]):
            if a and a < TEAM_ID_FLOOR and a not in names:
                names[a] = b

    # ---- home/away team per game ---------------------------------------------
    hd = df.dropna(subset=["HOMEDESCRIPTION"])
    home_team = hd.groupby("GAME_ID")["PLAYER1_TEAM_ID"].first().to_dict()
    game_teams = {}
    for col in ("PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID", "PLAYER3_TEAM_ID"):
        t = df[["GAME_ID", col]].dropna()
        t = t[t[col] >= TEAM_ID_FLOOR]
        for g, tid in zip(t.GAME_ID, t[col].astype("int64")):
            game_teams.setdefault(g, set()).add(tid)
    away_team = {}
    for g, ts in game_teams.items():
        h = home_team.get(g)
        others = [x for x in ts if x != h]
        if h is not None and len(others) == 1:
            away_team[g] = others[0]

    # ---- minutes + team per player (from stints) ------------------------------
    pmin = {}     # (player, team) -> seconds
    for g, hl, al, dur in zip(stints.GAME_ID, stints.HOME_LINEUP,
                              stints.AWAY_LINEUP, stints.DURATION_SECONDS):
        ht, at = home_team.get(g), away_team.get(g)
        for p in parse(hl):
            if ht is not None:
                pmin[(p, ht)] = pmin.get((p, ht), 0.0) + dur
        for p in parse(al):
            if at is not None:
                pmin[(p, at)] = pmin.get((p, at), 0.0) + dur
    # collapse to the team where the player logged the most minutes
    best = {}
    for (p, t), sec in pmin.items():
        if p not in best or sec > best[p][1]:
            best[p] = (t, sec)
    tot = {}
    for (p, t), sec in pmin.items():
        tot[p] = tot.get(p, 0.0) + sec
    players = pd.DataFrame({
        "PLAYER_ID": list(tot.keys()),
        "NAME": [names.get(p, "") for p in tot],
        "TEAM_ID": [best[p][0] for p in tot],
        "MINUTES": [tot[p] / 60.0 for p in tot],
    })

    # ---- actual team net rating ----------------------------------------------
    tpd, tposs = {}, {}
    for g, pm, ps in zip(stints.GAME_ID, stints.PLUS_MINUS, stints.POSS):
        ht, at = home_team.get(g), away_team.get(g)
        if ht is None or at is None:
            continue
        tpd[ht] = tpd.get(ht, 0.0) + pm; tpd[at] = tpd.get(at, 0.0) - pm
        tposs[ht] = tposs.get(ht, 0.0) + ps; tposs[at] = tposs.get(at, 0.0) + ps
    teams = pd.DataFrame({
        "TEAM_ID": list(tpd.keys()),
        "ABBR": [TEAM_ID_ABBR.get(t, str(t)) for t in tpd],
        "ACTUAL_NET": [tpd[t] / tposs[t] * 100.0 for t in tpd],
    })

    stints[["GAME_ID", "PERIOD", "HOME_LINEUP", "AWAY_LINEUP", "POSS", "Y",
            "DURATION_SECONDS", "PLUS_MINUS"]].to_csv(
        CACHE / f"stints_{season}.csv", index=False)
    players.to_csv(CACHE / f"players_{season}.csv", index=False)
    teams.to_csv(CACHE / f"teams_{season}.csv", index=False)
    os.remove(csv)   # free ~90MB
    print(f"season {season} (shuf {shuf}): {len(stints)} stints, "
          f"{len(players)} players, {len(teams)} teams, "
          f"net range [{teams.ACTUAL_NET.min():.1f}, {teams.ACTUAL_NET.max():.1f}]")


if __name__ == "__main__":
    for s in SEASONS:
        process_season(s)
    print("done")
