"""
Ingest the 2025-26 season (repo season 2026) via nba_api and build the cache
artifacts the backtest/outputs consume:

    cache/stints_2026.csv    GAME_ID, PERIOD, HOME_LINEUP, AWAY_LINEUP, POSS, Y,
                             DURATION_SECONDS, PLUS_MINUS
    cache/players_2026.csv   PLAYER_ID, NAME, TEAM_ID, MINUTES
    cache/teams_2026.csv     TEAM_ID, ABBR, ACTUAL_NET

Approach (two cheap calls per game, both cached + resumable):
  1. PlayByPlayV3 -> classic columns (pbpv3.convert).
  2. BoxScoreTraditionalV3 -> the 5 starters per team (position field) and a
     complete roster name->id map for the game.
We then walk substitutions chronologically from the Q1 starters (carrying the
lineup across period breaks, where between-quarter subs appear at 12:00), tagging
every event with the 10 on-court players, and segment stints at lineup changes.
This avoids the per-period boxscore calls that get rate-limited.

Requires Python 3.10+ with nba_api.
"""
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playbyplayv3, boxscoretraditionalv3
from nba_api.stats.static import players as static_players

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
PBP_CACHE = CACHE / "pbp_2026"
BOX_CACHE = CACHE / "box_2026"
STINT_CACHE = CACHE / "stints_2026_games"
for d in (PBP_CACHE, BOX_CACHE, STINT_CACHE):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
import pbpv3  # noqa: E402

warnings.filterwarnings("ignore")

SEASON = 2026
SEC_PER_POSS = 28.8
REQUEST_PAUSE = 0.7
SUB = 8
_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+", re.IGNORECASE)

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
ABBR_TO_ID = {v: k for k, v in TEAM_ID_ABBR.items()}
ABBR_TO_ID.update({"BKN": 1610612751, "PHX": 1610612756, "CHA": 1610612766})


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower()).strip()
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return re.sub(r"\s+", " ", s).strip()


def _retry(fn):
    for attempt in range(4):
        try:
            return fn()
        except Exception as exc:               # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt+1} after {wait}s: {exc}")
            time.sleep(wait)
    return None


def fetch_pbp(game_id):
    cache_file = PBP_CACHE / f"{game_id}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    df = _retry(lambda: playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=60).get_data_frames()[0])
    if df is not None:
        df.to_parquet(cache_file)
        time.sleep(REQUEST_PAUSE)
    return df


def fetch_box(game_id):
    cache_file = BOX_CACHE / f"{game_id}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    df = _retry(lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=60).get_data_frames()[0])
    if df is not None:
        df.to_parquet(cache_file)
        time.sleep(REQUEST_PAUSE)
    return df


def walk_lineups(classic, box, home_tid, away_tid):
    """Tag every event with the 10 on-court players by walking substitutions."""
    box = box.copy()
    box["pid"] = pd.to_numeric(box["personId"], errors="coerce")
    box = box.dropna(subset=["pid"])
    box["pid"] = box.pid.astype("int64")
    box["tid"] = pd.to_numeric(box["teamId"], errors="coerce").astype("int64")

    starters = {}
    name_to_id = {}
    for tid, g in box.groupby("tid"):
        st = g[g.position.astype(str).str.strip() != ""].pid.tolist()
        starters[tid] = set(st[:5])
        for pid, fn, ln in zip(g.pid, g.firstName, g.familyName):
            name_to_id.setdefault(_norm(ln), pid)
            name_to_id.setdefault(_norm(f"{fn} {ln}"), pid)
    if len(starters.get(home_tid, set())) != 5 or len(starters.get(away_tid, set())) != 5:
        return None

    df = classic.sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)
    cur = {home_tid: set(starters[home_tid]), away_tid: set(starters[away_tid])}
    home_rows, away_rows = [], []
    p1 = df.PLAYER1_ID.to_numpy()
    tcol = pd.to_numeric(df.PLAYER1_TEAM_ID, errors="coerce").to_numpy()
    et = df.EVENTMSGTYPE.to_numpy()
    desc = df.DESCRIPTION.to_numpy()
    for i in range(len(df)):
        home_rows.append(tuple(sorted(cur[home_tid])))
        away_rows.append(tuple(sorted(cur[away_tid])))
        if et[i] == SUB:
            tid = int(tcol[i]) if not np.isnan(tcol[i]) else None
            if tid not in cur:
                continue
            out_id = int(p1[i])
            m = _SUB_RE.search(str(desc[i]))
            in_id = name_to_id.get(_norm(m.group(1))) if m else None
            if in_id is None and m:
                key = _norm(m.group(1))
                for nm, pid in name_to_id.items():
                    if nm and (nm in key or key in nm):
                        in_id = pid
                        break
            if in_id is not None:
                cur[tid].discard(out_id)
                cur[tid].add(in_id)
    df["HOME5"] = home_rows
    df["AWAY5"] = away_rows
    return df


def stints_from_walk(df):
    df = df.copy()
    df["SCOREMARGIN"] = pd.to_numeric(df["SCOREMARGIN"], errors="coerce")
    rows = []
    for (gid, per), g in df.groupby(["GAME_ID", "PERIOD"], sort=True):
        g = g.sort_values("EVENTNUM").reset_index(drop=True)
        g["SCOREMARGIN"] = g["SCOREMARGIN"].ffill().bfill().fillna(0)
        sec = g["PCTIMESTRING"].map(_sec_remaining).to_numpy(dtype=float)
        marg = g["SCOREMARGIN"].to_numpy(dtype=float)
        home = g["HOME5"].tolist()
        away = g["AWAY5"].tolist()
        key = [h + a for h, a in zip(home, away)]
        starts = [0] + [i for i in range(1, len(g)) if key[i] != key[i - 1]]
        for s_idx, a in enumerate(starts):
            nxt = starts[s_idx + 1] if s_idx + 1 < len(starts) else None
            if nxt is not None:
                dur = sec[a] - sec[nxt]; pm = marg[nxt] - marg[a]
            else:
                dur = sec[a] - sec[len(g) - 1]; pm = marg[len(g) - 1] - marg[a]
            hl, al = list(home[a]), list(away[a])
            if dur <= 0 or len(hl) != 5 or len(al) != 5:
                continue
            rows.append({
                "GAME_ID": int(gid), "PERIOD": int(per),
                "HOME_LINEUP": ", ".join(map(str, hl)),
                "AWAY_LINEUP": ", ".join(map(str, al)),
                "DURATION_SECONDS": dur, "PLUS_MINUS": pm,
            })
    st = pd.DataFrame(rows)
    if st.empty:
        return st
    st["POSS"] = st.DURATION_SECONDS / SEC_PER_POSS
    st["Y"] = st.PLUS_MINUS / st.POSS * 100.0
    return st


def _sec_remaining(clock):
    m = re.match(r"(\d+):(\d+)", str(clock))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan


def process_game(game_id, home_tid, away_tid):
    cache_file = STINT_CACHE / f"{game_id}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    raw = fetch_pbp(game_id)
    box = fetch_box(game_id)
    if raw is None or not len(raw) or box is None or not len(box):
        return None
    classic = pbpv3.convert(raw)
    classic["GAME_ID"] = classic["GAME_ID"].astype("int64")
    walked = walk_lineups(classic, box, home_tid, away_tid)
    if walked is None:
        return None
    st = stints_from_walk(walked)
    st.to_parquet(cache_file)
    return st


def build_players_teams(stints, names, home_team, away_team):
    def parse(s):
        return [int(x) for x in str(s).split(",") if x.strip()]

    pmin = {}
    for g, hl, al, dur in zip(stints.GAME_ID, stints.HOME_LINEUP,
                              stints.AWAY_LINEUP, stints.DURATION_SECONDS):
        ht, at = home_team.get(g), away_team.get(g)
        for p in parse(hl):
            if ht is not None:
                pmin[(p, ht)] = pmin.get((p, ht), 0.0) + dur
        for p in parse(al):
            if at is not None:
                pmin[(p, at)] = pmin.get((p, at), 0.0) + dur
    best, tot = {}, {}
    for (p, t), sec in pmin.items():
        if p not in best or sec > best[p][1]:
            best[p] = (t, sec)
        tot[p] = tot.get(p, 0.0) + sec
    players = pd.DataFrame({
        "PLAYER_ID": list(tot.keys()),
        "NAME": [names.get(p, "") for p in tot],
        "TEAM_ID": [float(best[p][0]) for p in tot],
        "MINUTES": [tot[p] / 60.0 for p in tot],
    })

    tpd, tposs = {}, {}
    for g, pm, ps in zip(stints.GAME_ID, stints.PLUS_MINUS, stints.POSS):
        ht, at = home_team.get(g), away_team.get(g)
        if ht is None or at is None:
            continue
        tpd[ht] = tpd.get(ht, 0.0) + pm; tpd[at] = tpd.get(at, 0.0) - pm
        tposs[ht] = tposs.get(ht, 0.0) + ps; tposs[at] = tposs.get(at, 0.0) + ps
    teams = pd.DataFrame({
        "TEAM_ID": [float(t) for t in tpd],
        "ABBR": [TEAM_ID_ABBR.get(t, str(t)) for t in tpd],
        "ACTUAL_NET": [tpd[t] / tposs[t] * 100.0 for t in tpd],
    })
    return players, teams


def main():
    games = pd.read_csv(CACHE / f"games_{SEASON}.csv")
    names = {int(p["id"]): p["full_name"] for p in static_players.get_players()}
    home_team = {int(r.GAME_ID): float(ABBR_TO_ID[r.HOME]) for _, r in games.iterrows()}
    away_team = {int(r.GAME_ID): float(ABBR_TO_ID[r.AWAY]) for _, r in games.iterrows()}
    game_ids = [f"{int(g):010d}" for g in games.GAME_ID]
    print(f"processing {len(game_ids)} games (resumable cache in {STINT_CACHE.name}/) ...")

    all_st = []
    for i, gid in enumerate(game_ids, 1):
        gint = int(gid)
        st = process_game(gid, int(home_team[gint]), int(away_team[gint]))
        if st is not None and len(st):
            all_st.append(st)
        if i % 100 == 0:
            print(f"  {i}/{len(game_ids)} games, {sum(len(x) for x in all_st)} stints",
                  flush=True)

    stints = pd.concat(all_st, ignore_index=True)
    players, teams = build_players_teams(stints, names, home_team, away_team)
    stints[["GAME_ID", "PERIOD", "HOME_LINEUP", "AWAY_LINEUP", "POSS", "Y",
            "DURATION_SECONDS", "PLUS_MINUS"]].to_csv(
        CACHE / f"stints_{SEASON}.csv", index=False)
    players.to_csv(CACHE / f"players_{SEASON}.csv", index=False)
    teams.to_csv(CACHE / f"teams_{SEASON}.csv", index=False)
    clean = ((stints.HOME_LINEUP.str.count(",") == 4)
             & (stints.AWAY_LINEUP.str.count(",") == 4)).mean()
    print(f"season {SEASON}: {len(stints)} stints ({clean*100:.1f}% clean 5v5), "
          f"{len(players)} players, {len(teams)} teams, "
          f"net range [{teams.ACTUAL_NET.min():.1f}, {teams.ACTUAL_NET.max():.1f}]")


if __name__ == "__main__":
    main()
