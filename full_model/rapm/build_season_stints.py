"""
Offline lineup-stint reconstruction from shufinskiy nbastats play-by-play.

Reproduces the Dianjeol pipeline (quarter active players -> substitution patterns
-> quarter starters -> stints -> lineup walk) WITHOUT any live NBA API calls, so it
runs on any season's nbastats_{season}.csv. Output schema matches Dianjeol's
lineup_stints.csv:  GAME_ID, PERIOD, HOME_LINEUP, AWAY_LINEUP, DURATION_SECONDS,
PLUS_MINUS -- plus HOME_PTS / AWAY_PTS (real per-side points from the running
score) whenever the pbp carries it (classic `SCORE` = "AWAY - HOME", or
HOME_SCORE / AWAY_SCORE from data_ingest/pbpv3.convert).

Stint scoring convention (audit fix 2026-07): a stint's points are
    score after its LAST event  -  score after the PREVIOUS stint's last event,
with the running score forward-filled across the whole GAME (not per period).
The old version took `first(MARGIN)` of the stint as the start value -- i.e. the
margin AFTER the stint's first event -- and back-filled across period starts, so
any score on the first play after a substitution (typically the free throw after
a foul + sub) and the first score of every period was credited to NO lineup.
Stint PM then reproduced the final margin in only ~5% of games (MAE ~5 pts/game);
the fixed builder reconciles every game exactly (all stints, incl. non-5v5).
Duration uses the same convention: a stint runs from the clock of the PREVIOUS
stint's last event (the substitution that opened it; period start for the first
stint) to the clock of its own last event. The old START_SEC was the clock of the
stint's first logged event, so the dead-ball gap between every sub and the next
event was lost: ~9% of game time vanished (2625 of 2880 s/game), stint POSS were
under-counted, per-100 Y inflated (league offense read ~120 instead of ~114), and
stint-derived player minutes ran ~10% low.
Zero-duration stints are still dropped (no possessions to attribute).

Usage:
    from build_season_stints import build_lineup_stints
    df = pd.read_csv("pbp/nbastats_2022.csv", low_memory=False)
    stints = build_lineup_stints(df)
"""
import numpy as np
import pandas as pd

SUB = 8                  # EVENTMSGTYPE for substitution
TEAM_ID_FLOOR = 1610612000   # NBA team ids are >= this; player ids are far smaller


def _sec(t):
    if isinstance(t, str) and ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + int(s)
    return np.nan


def build_lineup_stints(df):
    df = df.copy()
    df["GAME_ID"] = df["GAME_ID"].astype("int64")
    df["PERIOD"] = df["PERIOD"].astype(int)

    # ---- home team per game (team on the home description side) ---------------
    hd = df.dropna(subset=["HOMEDESCRIPTION"])
    home_team = hd.groupby("GAME_ID")["PLAYER1_TEAM_ID"].first().to_dict()

    # ---- long player/team table (valid players only) -------------------------
    longs = []
    for pcol, tcol in [("PLAYER1_ID", "PLAYER1_TEAM_ID"),
                       ("PLAYER2_ID", "PLAYER2_TEAM_ID"),
                       ("PLAYER3_ID", "PLAYER3_TEAM_ID")]:
        part = df[["GAME_ID", "PERIOD", pcol, tcol]].rename(
            columns={pcol: "PID", tcol: "TID"})
        longs.append(part)
    lp = pd.concat(longs, ignore_index=True).dropna(subset=["PID"])
    lp = lp[(lp.PID != 0) & (lp.PID < TEAM_ID_FLOOR)]
    lp["PID"] = lp.PID.astype("int64")

    # player -> team per game (mode of non-null team ids)
    pt = lp.dropna(subset=["TID"])
    pt = pt[pt.TID >= TEAM_ID_FLOOR]
    player_team = (pt.groupby(["GAME_ID", "PID"])["TID"]
                     .agg(lambda s: s.value_counts().index[0]).to_dict())

    def role(gid, pid):
        t = player_team.get((gid, pid))
        if t is None:
            return None
        return "home" if t == home_team.get(gid) else "away"

    # active players per (game, period) split by role
    active = lp.drop_duplicates(["GAME_ID", "PERIOD", "PID"])[["GAME_ID", "PERIOD", "PID"]]
    active["ROLE"] = [role(g, p) for g, p in zip(active.GAME_ID, active.PID)]
    active = active.dropna(subset=["ROLE"])

    # ---- substitutions log ----------------------------------------------------
    subs = df[df.EVENTMSGTYPE == SUB][["GAME_ID", "PERIOD", "PCTIMESTRING",
                                       "PLAYER1_ID", "PLAYER2_ID"]].copy()
    subs = subs[(subs.PLAYER1_ID != 0) & (subs.PLAYER2_ID != 0)]
    subs["SEC"] = subs.PCTIMESTRING.map(_sec)
    subs["PLAYER1_ID"] = subs.PLAYER1_ID.astype("int64")
    subs["PLAYER2_ID"] = subs.PLAYER2_ID.astype("int64")

    # ---- quarter starters: active minus players whose first sub event is IN ---
    # build per (game,period,player) first sub event type, chronological (sec desc)
    out_ev = subs[["GAME_ID", "PERIOD", "SEC", "PLAYER1_ID"]].rename(
        columns={"PLAYER1_ID": "PID"})
    out_ev["TYPE"] = "OUT"
    in_ev = subs[["GAME_ID", "PERIOD", "SEC", "PLAYER2_ID"]].rename(
        columns={"PLAYER2_ID": "PID"})
    in_ev["TYPE"] = "IN"
    ev = pd.concat([out_ev, in_ev], ignore_index=True)
    # chronological order within a period = descending seconds remaining;
    # stable so ties keep log order
    ev = ev.sort_values(["GAME_ID", "PERIOD", "PID", "SEC"],
                        ascending=[True, True, True, False])
    first = ev.groupby(["GAME_ID", "PERIOD", "PID"]).TYPE.first().reset_index()
    non_starters = set(map(tuple, first[first.TYPE == "IN"][
        ["GAME_ID", "PERIOD", "PID"]].itertuples(index=False, name=None)))

    starters = {}   # (game, period) -> {'home': set, 'away': set}
    for g, per, pid, r in zip(active.GAME_ID, active.PERIOD, active.PID, active.ROLE):
        if (g, per, pid) in non_starters:
            continue
        d = starters.setdefault((g, per), {"home": set(), "away": set()})
        d[r].add(pid)

    # ---- stints: boundaries at subs and period change -------------------------
    keep = ["GAME_ID", "PERIOD", "PCTIMESTRING", "EVENTMSGTYPE", "SCOREMARGIN", "EVENTNUM"]
    keep += [c for c in ("SCORE", "HOME_SCORE", "AWAY_SCORE") if c in df.columns]
    s = df[keep].copy()
    s["SEC"] = s.PCTIMESTRING.map(_sec)
    s = s.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)
    has_pts = True
    if "HOME_SCORE" in s.columns and "AWAY_SCORE" in s.columns:
        s["H_SC"] = pd.to_numeric(s.HOME_SCORE, errors="coerce")
        s["A_SC"] = pd.to_numeric(s.AWAY_SCORE, errors="coerce")
    elif "SCORE" in s.columns:
        sc = s.SCORE.astype(str).str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
        s["A_SC"], s["H_SC"] = sc[0], sc[1]            # classic SCORE is "AWAY - HOME"
    else:
        has_pts = False
    if has_pts:
        # running score is cumulative over the game: ffill across periods, 0 at tip
        for c in ("H_SC", "A_SC"):
            s[c] = s.groupby("GAME_ID")[c].ffill().fillna(0.0)
        s["MARGIN"] = s.H_SC - s.A_SC
    else:
        s["MARGIN"] = pd.to_numeric(s.SCOREMARGIN.replace("TIE", 0), errors="coerce")
        s["MARGIN"] = s.groupby("GAME_ID").MARGIN.ffill().fillna(0.0)
        s["H_SC"] = np.nan; s["A_SC"] = np.nan
    # value BEFORE each event = value after the previous event of the same game
    s["MARGIN_PREV"] = s.groupby("GAME_ID").MARGIN.shift(1).fillna(0.0)
    s["H_PREV"] = s.groupby("GAME_ID").H_SC.shift(1).fillna(0.0)
    s["A_PREV"] = s.groupby("GAME_ID").A_SC.shift(1).fillna(0.0)
    s["IS_SUB"] = s.EVENTMSGTYPE == SUB
    s["STINT_ENDS"] = s.IS_SUB.shift(1, fill_value=False)
    s["PERIOD_CHANGE"] = (s.PERIOD != s.PERIOD.shift(1)) | (s.GAME_ID != s.GAME_ID.shift(1))
    s["BOUNDARY"] = s.STINT_ENDS | s.PERIOD_CHANGE
    s["STINT_ID"] = s.BOUNDARY.cumsum()

    st = s.groupby("STINT_ID").agg(
        GAME_ID=("GAME_ID", "first"),
        PERIOD=("PERIOD", "first"),
        FIRST_EV_SEC=("SEC", "first"),
        END_SEC=("SEC", "last"),
        START_MARGIN=("MARGIN_PREV", "first"),
        END_MARGIN=("MARGIN", "last"),
        H0=("H_PREV", "first"), H1=("H_SC", "last"),
        A0=("A_PREV", "first"), A1=("A_SC", "last"),
    ).reset_index()
    # stint starts when the previous stint (same game+period) ended; the first stint
    # of a period starts at the period's opening clock (12:00 regulation, 5:00 OT)
    prev_end = st.groupby(["GAME_ID", "PERIOD"]).END_SEC.shift(1)
    period_len = np.where(st.PERIOD <= 4, 720.0, 300.0)
    st["START_SEC"] = prev_end.fillna(pd.Series(period_len, index=st.index))
    st["START_SEC"] = np.maximum(st.START_SEC, st.FIRST_EV_SEC.fillna(st.START_SEC))
    st["DURATION_SECONDS"] = st.START_SEC - st.END_SEC
    st["PLUS_MINUS"] = st.END_MARGIN - st.START_MARGIN
    st["HOME_PTS"] = (st.H1 - st.H0) if has_pts else np.nan
    st["AWAY_PTS"] = (st.A1 - st.A0) if has_pts else np.nan
    st = st[st.DURATION_SECONDS > 0].copy()

    # ---- lineup walk per (game, period) ---------------------------------------
    sub_at = {}    # (game, period, sec) -> list of (out, in)
    for g, per, sec, po, pi in zip(subs.GAME_ID, subs.PERIOD, subs.SEC,
                                   subs.PLAYER1_ID, subs.PLAYER2_ID):
        sub_at.setdefault((g, per, sec), []).append((po, pi))

    rows = []
    for (g, per), grp in st.groupby(["GAME_ID", "PERIOD"]):
        info = starters.get((g, per), {"home": set(), "away": set()})
        home_lu = set(info["home"]); away_lu = set(info["away"])
        for _, stint in grp.sort_values("START_SEC", ascending=False).iterrows():
            rows.append({
                "GAME_ID": g, "PERIOD": per,
                "HOME_LINEUP": ", ".join(map(str, sorted(home_lu))),
                "AWAY_LINEUP": ", ".join(map(str, sorted(away_lu))),
                "DURATION_SECONDS": stint.DURATION_SECONDS,
                "PLUS_MINUS": stint.PLUS_MINUS,
                "HOME_PTS": stint.HOME_PTS, "AWAY_PTS": stint.AWAY_PTS,
                "START_SEC": stint.START_SEC, "END_SEC": stint.END_SEC,
            })
            for po, pi in sub_at.get((g, per, stint.END_SEC), []):
                r = role(g, po)
                if r == "home":
                    home_lu.discard(po); home_lu.add(pi)
                elif r == "away":
                    away_lu.discard(po); away_lu.add(pi)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    df = pd.read_csv(f"pbp/nbastats_{season}.csv", low_memory=False)
    out = build_lineup_stints(df)
    out.to_csv(f"lineup_stints_{season}.csv", index=False)
    n5 = ((out.HOME_LINEUP.str.count(",") == 4) & (out.AWAY_LINEUP.str.count(",") == 4)).mean()
    print(f"season {season}: {len(out)} stints, {n5*100:.1f}% are clean 5v5")
