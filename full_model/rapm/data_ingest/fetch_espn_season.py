"""
In-season ingest from ESPN game summaries (stats.nba.com / cdn.nba.com are 403 here).

For every COMPLETED regular-season game in cache/games_{s}.csv (needs ESPN_ID, written by
fetch_schedule.py) this fetches the ESPN summary (cached as JSON, resumable) and writes:

  cache/games_{s}.csv        HOME_PTS / AWAY_PTS / HOME_WIN filled for finished games
  cache/stints_{s}.csv       5v5 stints with REAL per-side points (same schema/conventions
                             as fetch_nba_2026: walk substitutions from the box-score starters,
                             re-seed each period >= 2 from in-period activity because
                             between-period subs are not logged, synthetic period-start rows)
  cache/espn_box_{s}.csv     per player-game box line (fantasy scoring, sanity checks)

players_{s}.csv is NOT touched: for the live season it holds the projected roster / minutes
that every forecast consumer reads. ESPN athlete ids map to BOOKER PLAYER_IDs by name against
players_{s}.csv first (rebuild_rosters_2027 already used ESPN ids for players unknown to NBA
ids), then earlier seasons; unmatched players keep their ESPN id.

Usage: python data_ingest/fetch_espn_season.py 2027
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
SEC_PER_POSS = 28.8
SUB = 8
ABBR_FIX = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
            "BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}
TEAM_ID = {"ATL": 1610612737, "BOS": 1610612738, "CLE": 1610612739, "NOP": 1610612740, "CHI": 1610612741,
           "DAL": 1610612742, "DEN": 1610612743, "GSW": 1610612744, "HOU": 1610612745, "LAC": 1610612746,
           "LAL": 1610612747, "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750, "BRK": 1610612751,
           "NYK": 1610612752, "ORL": 1610612753, "IND": 1610612754, "PHI": 1610612755, "PHO": 1610612756,
           "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "OKC": 1610612760, "TOR": 1610612761,
           "UTA": 1610612762, "MEM": 1610612763, "WAS": 1610612764, "DET": 1610612765, "CHO": 1610612766}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv", "v"})


def _get(url, tries=4):
    for k in range(tries):
        r = subprocess.run(["curl", "-s", "--compressed", "-m", "30", url], capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None


def summary(season, eid):
    d = CACHE / f"espn_games_{season}"
    d.mkdir(exist_ok=True)
    p = d / f"{eid}.json"
    if p.exists():
        return json.loads(p.read_text())
    j = _get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={eid}")
    if j and j.get("header", {}).get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed"):
        p.write_text(json.dumps(j))          # cache only finished games
    return j


def _clock(s):
    s = str(s)
    if ":" in s:
        m, x = s.split(":")
        return int(m) * 60 + int(float(x))
    return int(float(s))


class IdMap:
    """ESPN athlete -> BOOKER PLAYER_ID by normalised name (live roster first)."""
    def __init__(self, season):
        self.by_name = {}
        for s in range(season, 2014, -1):
            p = CACHE / f"players_{s}.csv"
            if p.exists():
                pl = pd.read_csv(p)
                for pid, nm in zip(pl.PLAYER_ID.astype(int), pl.NAME):
                    self.by_name.setdefault(_norm(nm), pid)
        self.miss = set()

    def __call__(self, eid, name):
        pid = self.by_name.get(_norm(name))
        if pid is None:
            self.miss.add(name)
            return int(eid)
        return int(pid)


def parse_game(j, gid, idmap):
    comp = j["header"]["competitions"][0]
    side = {c["team"]["id"]: c["homeAway"] for c in comp["competitors"]}
    abbr = {c["team"]["id"]: ABBR_FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"]) for c in comp["competitors"]}
    score = {c["homeAway"]: int(c["score"]) for c in comp["competitors"]}
    ath, box, starters = {}, [], {"home": set(), "away": set()}
    for t in j["boxscore"]["players"]:
        ha = side[t["team"]["id"]]
        st = t["statistics"][0]
        names = st.get("names") or st.get("labels")
        for a in st["athletes"]:
            pid = idmap(a["athlete"]["id"], a["athlete"]["displayName"])
            ath[a["athlete"]["id"]] = (pid, ha)
            if a.get("starter"):
                starters[ha].add(pid)
            vals = a.get("stats") or []
            if len(vals) == len(names):
                r = dict(zip(names, vals))
                def mk(x):
                    m, a_ = (str(r.get(x, "0-0")) + "-0").split("-")[:2]
                    return int(m or 0), int(a_ or 0)
                fgm, fga = mk("FG"); tpm, tpa = mk("3PT"); ftm, fta = mk("FT")
                num = lambda k: int(float(r.get(k, 0) or 0)) if str(r.get(k, "")).lstrip("-").replace(".", "").isdigit() else 0
                box.append({"GAME_ID": gid, "PLAYER_ID": pid, "NAME": a["athlete"]["displayName"],
                            "TEAM": abbr[t["team"]["id"]], "starter": int(bool(a.get("starter"))),
                            "min": num("MIN"), "pts": num("PTS"), "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa,
                            "ftm": ftm, "fta": fta, "reb": num("REB"), "ast": num("AST"), "stl": num("STL"),
                            "blk": num("BLK"), "tov": num("TO"), "oreb": num("OREB"), "dreb": num("DREB"),
                            "pf": num("PF")})
    if len(starters["home"]) != 5 or len(starters["away"]) != 5:
        return None, box, score
    rows, seen_period = [], set()
    for k, p in enumerate(j.get("plays", [])):
        per = int(p["period"]["number"])
        if per not in seen_period:                 # synthetic period-start row (12:00 / 5:00)
            seen_period.add(per)
            rows.append({"PERIOD": per, "EVENTNUM": 2 * k, "SEC": 720 if per <= 4 else 300,
                         "TYPE": 0, "P1": 0, "P2": 0, "SIDE": None,
                         "HOME_SCORE": np.nan, "AWAY_SCORE": np.nan})
        parts = [x["athlete"]["id"] for x in p.get("participants", []) if "athlete" in x]
        tsd = side.get(p.get("team", {}).get("id"))
        is_sub = p["type"]["text"] == "Substitution" or "enters the game for" in p.get("text", "")
        p1 = ath.get(parts[0], (0, None))[0] if parts else 0
        p2 = ath.get(parts[1], (0, None))[0] if len(parts) > 1 else 0
        if is_sub:                                  # ESPN: [IN, OUT] -> P1 = OUT, P2 = IN
            p1, p2 = p2, p1
        rows.append({"PERIOD": per, "EVENTNUM": 2 * k + 1, "SEC": _clock(p["clock"]["displayValue"]),
                     "TYPE": SUB if is_sub else 1, "P1": p1, "P2": p2,
                     "SIDE": tsd if tsd else (ath.get(parts[0], (0, None))[1] if parts else None),
                     "HOME_SCORE": p.get("homeScore"), "AWAY_SCORE": p.get("awayScore")})
    ev = pd.DataFrame(rows).sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)
    roster = {"home": {v[0] for v in ath.values() if v[1] == "home"},
              "away": {v[0] for v in ath.values() if v[1] == "away"}}
    side_of = {v[0]: v[1] for v in ath.values()}
    return walk(ev, starters, roster, side_of), box, score


def walk(ev, starters, roster, side_of):
    """Tag each event with the 10 on-court players (cf. fetch_nba_2026.walk_lineups)."""
    cur = {"home": set(starters["home"]), "away": set(starters["away"])}
    per = ev.PERIOD.to_numpy(); typ = ev.TYPE.to_numpy(); p1 = ev.P1.to_numpy(); p2 = ev.P2.to_numpy()

    def period_starters(lo, hi, carried):
        seen, first_in = {}, set()
        for i in range(lo, hi):
            if typ[i] == SUB:
                o, n = int(p1[i]), int(p2[i])
                if o and o not in seen:
                    seen[o] = "start"
                if n and n not in seen:
                    seen[n] = "in"; first_in.add(n)
            else:
                for q in (int(p1[i]), int(p2[i])):
                    if q and q not in seen and q in side_of:
                        seen[q] = "start"
        out = {}
        for sd in ("home", "away"):
            st = [q for q, k in seen.items() if k == "start" and side_of.get(q) == sd]
            if len(st) > 5:
                st = sorted(st, key=lambda q: q not in carried[sd])[:5]
            for q in sorted(carried[sd]):
                if len(st) >= 5:
                    break
                if q not in st and q not in first_in:
                    st.append(q)
            out[sd] = set(st) if len(st) == 5 else set(carried[sd])
        return out

    bounds = {}
    for i in range(len(ev)):
        bounds.setdefault(int(per[i]), [i, i])[1] = i + 1
    H, A = [], []
    for i in range(len(ev)):
        if i > 0 and per[i] != per[i - 1]:
            lo, hi = bounds[int(per[i])]
            cur = period_starters(lo, hi, cur)
        H.append(tuple(sorted(cur["home"]))); A.append(tuple(sorted(cur["away"])))
        if typ[i] == SUB:
            o, n = int(p1[i]), int(p2[i])
            sd = side_of.get(o) or side_of.get(n)
            if sd in cur and o and n:
                cur[sd].discard(o); cur[sd].add(n)
    ev = ev.copy()
    ev["HOME5"], ev["AWAY5"] = H, A
    return ev


def stints(ev, gid):
    """Segment a walked game into stints with real per-side points (fetch_nba_2026 rules)."""
    ev = ev.copy()
    ev["_H"] = pd.to_numeric(ev.HOME_SCORE, errors="coerce").ffill().fillna(0.0)
    ev["_A"] = pd.to_numeric(ev.AWAY_SCORE, errors="coerce").ffill().fillna(0.0)
    ev["_H0"] = ev._H.shift(1).fillna(0.0); ev["_A0"] = ev._A.shift(1).fillna(0.0)
    rows = []
    for p_, g in ev.groupby("PERIOD", sort=True):
        g = g.reset_index(drop=True)
        sec = g.SEC.to_numpy(float)
        h1, a1, h0, a0 = g._H.to_numpy(), g._A.to_numpy(), g._H0.to_numpy(), g._A0.to_numpy()
        key = [h + a for h, a in zip(g.HOME5, g.AWAY5)]
        starts = [0] + [i for i in range(1, len(g)) if key[i] != key[i - 1]]
        for s_idx, a in enumerate(starts):
            nxt = starts[s_idx + 1] if s_idx + 1 < len(starts) else len(g)
            last = nxt - 1
            dur = (sec[a - 1] if a > 0 else sec[0]) - sec[last]
            hl, al = list(g.HOME5[a]), list(g.AWAY5[a])
            if dur <= 0 or len(hl) != 5 or len(al) != 5:
                continue
            hp, ap = h1[last] - h0[a], a1[last] - a0[a]
            rows.append({"GAME_ID": gid, "PERIOD": int(p_), "HOME_LINEUP": ", ".join(map(str, hl)),
                         "AWAY_LINEUP": ", ".join(map(str, al)), "DURATION_SECONDS": dur,
                         "PLUS_MINUS": hp - ap, "HOME_PTS": hp, "AWAY_PTS": ap})
    st = pd.DataFrame(rows)
    if len(st):
        st["POSS"] = st.DURATION_SECONDS / SEC_PER_POSS
        st["Y"] = st.PLUS_MINUS / st.POSS * 100.0
    return st


def main(season):
    gp = CACHE / f"games_{season}.csv"
    G = pd.read_csv(gp)
    idmap = IdMap(season)
    today = pd.Timestamp.now(tz="US/Eastern").strftime("%Y-%m-%d")
    todo = G[(G.SEASON_TYPE == "Regular Season") & G.ESPN_ID.notna() & (G.DATE <= today)]
    all_st, all_box, done = [], [], 0
    for r in todo.itertuples():
        j = summary(season, int(r.ESPN_ID))
        if not j or not j["header"]["competitions"][0]["status"]["type"].get("completed"):
            continue
        ev, box, score = parse_game(j, int(r.GAME_ID), idmap)
        all_box += box
        i = G.index[G.GAME_ID == r.GAME_ID][0]
        G.at[i, "HOME_PTS"], G.at[i, "AWAY_PTS"] = score["home"], score["away"]
        G.at[i, "HOME_WIN"] = int(score["home"] > score["away"])
        done += 1
        if ev is None:
            print(f"  {r.DATE} {r.AWAY}@{r.HOME}: starters missing, no stints"); continue
        st = stints(ev, int(r.GAME_ID))
        sh, sa = st.HOME_PTS.sum(), st.AWAY_PTS.sum()
        mins = st.DURATION_SECONDS.sum() / 60
        print(f"  {r.DATE} {r.AWAY}@{r.HOME} {score['away']}-{score['home']}: {len(st)} stints, "
              f"{mins:.0f} min, stint pts {sa:.0f}-{sh:.0f}", flush=True)
        all_st.append(st)
    G.to_csv(gp, index=False)
    if all_st:
        S = pd.concat(all_st, ignore_index=True)
        S[["GAME_ID", "PERIOD", "HOME_LINEUP", "AWAY_LINEUP", "POSS", "Y", "DURATION_SECONDS",
           "PLUS_MINUS", "HOME_PTS", "AWAY_PTS"]].to_csv(CACHE / f"stints_{season}.csv", index=False)
        pd.DataFrame(all_box).to_csv(CACHE / f"espn_box_{season}.csv", index=False)
        sys.path.insert(0, str(RAPM))
        from stint_off_def import enrich_season          # Y_OFF_HOME / Y_DEF_HOME from real points
        enrich_season(season)
        print(f"season {season}: {done} completed games, {len(S)} stints, {len(all_box)} player box lines")
    if idmap.miss:
        print(f"  {len(idmap.miss)} players kept ESPN ids (no BOOKER match): {sorted(idmap.miss)[:12]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2027)
