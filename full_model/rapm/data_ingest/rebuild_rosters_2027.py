"""Rebuild cache/players_2027.csv from LIVE ESPN rosters + the gated minutes model.

(2026-10) With --write, the roster is then re-projected by forecast.minutes_share --
the walk-forward minutes-SHARE model (previous season, rating, roster fit/depth, age),
which overwrites MINUTES (overrides respected). The legacy minutes ridge below only
seeds the file; the share model is what every team aggregate consumes.

Rosters: ESPN team roster API (current as of run date -- catches FA/trades the
static file misses). Names are mapped to BOOKER PLAYER_IDs via norm_name over
every players_{s}.csv + the ratings CSV; unmapped players (new draftees) carry
their ESPN athlete id (>=3M, no collision with NBA ids <=2M) and a default.

Minutes: walk-forward-gated model (scratch minutes_gate.py, panel = 2,038
transitions 2018-2024 targets):
    ridge on [h0,h1,h2,mx,prior_mx,quality,age]   RMSE 462 (rule 497, naive 521)
    injury class (h0 < 45% of prior-2yr max >=1500): 50/50 blend with the old
    recovery rule (ridge 512 / rule 479 / blend ~480 there; young 541 -> 461).
h0 = FULL 2026 minutes from cache/espn_pbp_2026.csv (the stint feed captured
only ~72% of 2026); h1/h2 from players_2025/2024. Rookies keep old defaults.

Run: python data_ingest/rebuild_rosters_2027.py [--write]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
sys.path.insert(0, str(RAPM))
from forecast import player_impacts as pi          # noqa: E402
from forecast import uncertainty as unc            # noqa: E402

H = {"User-Agent": "Mozilla/5.0"}
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster"
ABBR_FIX = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
            "BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}   # basketball-reference-style codes


def fetch_rosters():
    js = requests.get(TEAMS_URL, headers=H, timeout=30).json()
    teams = [t["team"] for t in js["sports"][0]["leagues"][0]["teams"]]
    out = []
    for t in teams:
        ab = ABBR_FIX.get(t["abbreviation"], t["abbreviation"])
        r = requests.get(ROSTER_URL.format(tid=t["id"]), headers=H, timeout=30).json()
        for a in r.get("athletes", []):
            out.append({"espn_id": int(a["id"]), "name": a["fullName"], "abbr": ab})
        time.sleep(0.4)
    return pd.DataFrame(out)


def name_to_pid():
    m = {}
    for s in range(2020, 2028):
        p = CACHE / f"players_{s}.csv"
        if p.exists():
            d = pd.read_csv(p)
            for pid, nm in zip(d.PLAYER_ID, d.NAME):
                m[pi.norm_name(nm)] = int(pid)
    bk = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    for pid, nm in zip(bk.PLAYER_ID, bk.player):
        m[pi.norm_name(nm)] = int(pid)
    return m


def fit_minutes_model():
    """Ridge fit on the full transition panel; returns (model, medians, rule_fn)."""
    from sklearn.linear_model import Ridge
    bk = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    bk = bk[bk.season <= 2026]
    m = bk.pivot_table(index="PLAYER_ID", columns="season", values="minutes", aggfunc="first")
    sc = bk.pivot_table(index="PLAYER_ID", columns="season", values="booker_score", aggfunc="first")
    a = unc.attach_attrs(bk[["player", "season"]].assign(minutes=bk.minutes))
    ag = bk.assign(age=a.age.values).pivot_table(index="PLAYER_ID", columns="season",
                                                 values="age", aggfunc="first")
    rows = []
    for t in range(2017, 2025):                     # targets 2018-2025 (2026 partial: excluded)
        for pid in m.index:
            y = m.at[pid, t + 1] if (t + 1) in m.columns else np.nan
            h0 = m.at[pid, t] if t in m.columns else np.nan
            if not (y == y and h0 == h0):
                continue
            rows.append({
                "y": y, "h0": h0,
                "h1": m.at[pid, t - 1] if (t - 1) in m.columns else np.nan,
                "h2": m.at[pid, t - 2] if (t - 2) in m.columns else np.nan,
                "q": sc.at[pid, t] if t in sc.columns else np.nan,
                "age": ag.at[pid, t] if t in ag.columns else np.nan})
    D = pd.DataFrame(rows)
    D["mx"] = D[["h0", "h1", "h2"]].max(axis=1)
    D["prior_mx"] = D[["h1", "h2"]].max(axis=1)
    FE = ["h0", "h1", "h2", "mx", "prior_mx", "q", "age"]
    med = D[FE].median()
    mdl = Ridge(alpha=1.0).fit(D[FE].fillna(med).fillna(0), D.y)
    return mdl, med, FE


W = np.array([0.55, 0.30, 0.15])


def recovery_rule(h0, h1, h2, age):
    h = [h0, h1, h2]
    w = np.array([W[i] for i in range(3) if h[i] == h[i]])
    v = np.array([x for x in h if x == x])
    p = float((w / w.sum()) @ v)
    p = max(p, 0.50 * np.nanmax(h))
    if age == age:
        if age <= 23:
            p *= 1.10
        elif age >= 36:
            p *= 0.82
        elif age >= 33:
            p *= 0.90
    return p


def main(write=False):
    ros = fetch_rosters()
    print(f"ESPN rosters: {len(ros)} players, {ros.abbr.nunique()} teams")
    n2p = name_to_pid()
    ros["nm"] = ros.name.map(pi.norm_name)
    ros["pid"] = ros.nm.map(n2p)
    unmapped = ros[ros.pid.isna()]
    print(f"unmapped (new draftees / two-ways): {len(unmapped)}")
    ros["pid"] = ros.pid.fillna(ros.espn_id).astype(int)

    data = pi.BookerData(seasons=range(2024, 2027))
    abbr2tid = {ab: int(tid) for tid, ab in zip(data.TEAMS[2026].TEAM_ID, data.TEAMS[2026].ABBR)}

    espn = pd.read_csv(CACHE / "espn_pbp_2026.csv")
    m26 = dict(zip(espn.PLAYER_ID.astype(int), espn.minutes.astype(float)))
    m25 = dict(zip(*[pd.read_csv(CACHE / "players_2025.csv")[c] for c in ("PLAYER_ID", "MINUTES")]))
    m24 = dict(zip(*[pd.read_csv(CACHE / "players_2024.csv")[c] for c in ("PLAYER_ID", "MINUTES")]))
    p26 = pd.read_csv(CACHE / "players_2026.csv")
    m26_part = dict(zip(p26.PLAYER_ID.astype(int), p26.MINUTES.astype(float)))
    old = pd.read_csv(CACHE / "players_2027.csv")
    old_min = dict(zip(old.PLAYER_ID.astype(int), old.MINUTES.astype(float)))
    old_team = dict(zip(old.PLAYER_ID.astype(int), old.TEAM_ID.astype(int)))
    bk = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    bk26 = bk[bk.season == 2026].set_index("PLAYER_ID")     # rating THROUGH 2026
    q_map = bk26.booker_score.to_dict()
    a = unc.attach_attrs(bk.drop_duplicates("PLAYER_ID", keep="last")[["player", "season"]]
                         .assign(minutes=0))
    age_map = {}
    ref = bk.drop_duplicates("PLAYER_ID", keep="last")
    for pid, s0, ag in zip(ref.PLAYER_ID, ref.season, a.age.values):
        if ag == ag:
            age_map[int(pid)] = float(ag) + (2026 - int(s0))   # age as of feature season 2026

    mdl, med, FE = fit_minutes_model()
    rows, moves = [], []
    for r in ros.itertuples():
        pid = int(r.pid)
        h0 = m26.get(pid, m26_part.get(pid, np.nan))
        h1 = m25.get(pid, np.nan)
        h2 = m24.get(pid, np.nan)
        if h0 != h0 and h1 != h1 and h2 != h2:      # no NBA history: rookie default
            mins = old_min.get(pid, 400.0)
        else:
            hh = [h0, h1, h2]
            mx = np.nanmax(hh)
            pmx = np.nanmax([h1, h2]) if (h1 == h1 or h2 == h2) else np.nan
            q = q_map.get(pid, np.nan)
            age = age_map.get(pid, np.nan)
            X = pd.DataFrame([{"h0": h0 if h0 == h0 else 0.0, "h1": h1, "h2": h2,
                               "mx": mx, "prior_mx": pmx, "q": q, "age": age}])
            pred = float(mdl.predict(X[FE].fillna(med).fillna(0))[0])
            inj = (h0 == h0 and pmx == pmx and pmx >= 1500 and h0 < 0.45 * pmx)
            if inj:
                pred = 0.5 * pred + 0.5 * recovery_rule(h0, h1, h2, age)
            mins = float(np.clip(pred, 200, 3100))
        tid = abbr2tid.get(r.abbr)
        if tid is None:
            continue
        rows.append({"PLAYER_ID": pid, "NAME": r.name, "TEAM_ID": tid,
                     "MINUTES": round(mins, 1)})
        if pid in old_team and old_team[pid] != tid:
            moves.append((r.name, old_team[pid], tid))
    out = pd.DataFrame(rows).drop_duplicates("PLAYER_ID")
    # known-situation overrides (injury-recovery status, retirement plans -- news
    # the history-based model structurally cannot see; same category as the
    # market's informational edge in game odds). Kept small and documented.
    ovp = CACHE / "minutes_overrides_2027.csv"
    if ovp.exists():
        for r in pd.read_csv(ovp).itertuples():
            m = out.NAME == r.NAME
            if m.any():
                out.loc[m, "MINUTES"] = float(r.MINUTES)
                print(f"  override: {r.NAME} -> {r.MINUTES} ({r.REASON[:60]}...)")
    print(f"\nnew players_2027: {len(out)} rows | team moves vs old file: {len(moves)}")
    tid2ab = {v: k for k, v in abbr2tid.items()}
    for nm, a_, b_ in moves[:25]:
        print(f"  {nm}: {tid2ab.get(a_,'?')} -> {tid2ab.get(b_,'?')}")
    dropped = set(old.PLAYER_ID.astype(int)) - set(out.PLAYER_ID)
    print(f"dropped from league (not on any ESPN roster): {len(dropped)}")
    for nm in ("Jayson Tatum", "Victor Wembanyama", "LeBron James", "Nikola Jokic",
               "Kevin Durant", "Paul George"):
        rr = out[out.NAME == nm]
        if len(rr):
            print(f"  spot {nm}: {tid2ab.get(int(rr.TEAM_ID.iloc[0]))} {float(rr.MINUTES.iloc[0]):.0f} min")
        else:
            print(f"  spot {nm}: NOT ON A ROSTER")
    if write:
        (CACHE / "players_2027.csv").rename(CACHE / "players_2027.csv.bak2")
        out.to_csv(CACHE / "players_2027.csv", index=False)
        print("WROTE cache/players_2027.csv (old -> .bak2)")
        from forecast import minutes_share as ms
        ms.build()                                  # projected minutes shares (+ overrides)
    return out


if __name__ == "__main__":
    main(write="--write" in sys.argv)
