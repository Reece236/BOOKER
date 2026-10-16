"""Split each player's DEFENSE into INTERIOR and PERIMETER components.

1. Opponent scoring in every 5v5 stint, relative to league expectation, is decomposed
   EXACTLY (each shot from cache/shotdetail_{s}.csv mapped to the stint by game/period/clock;
   e_in / e_out = league points per interior / perimeter attempt, sh = league interior share):
       interior   = (pts_in - n_in*e_in)                 rim/paint efficiency allowed
                  + (n_in - sh*N) * (e_in - e_out)       DETERRENCE: attempts kept out of the
                                                         paint, net of what they turn into
                  + (FT - ft_rate*poss)                  free throws allowed (fouls)
       perimeter  = (pts_out - n_out*e_out)              jumper/three efficiency allowed
       possession = (N - fga_rate*poss) * e_bar          shot volume (TOs forced, DREB)
   A first version split raw points by zone; it credited "playing without a big" as
   perimeter defense (opponents' shots get pushed outward by rim protectors: corr(int, per)
   = -0.53, perimeter leaders Naz Reid / Maxey). Netting displacement into interior fixes it.
2. One prior-free ridge RAPM per component on the same 3-season decay windows as the
   BookerFormer ratings (offense + defense columns, so a defender is credited net of the
   shooters he faced): d_k = opponent points/100 SUPPRESSED in zone k (+ = good).
3. Reconciled to the rating: the components are shifted so they sum EXACTLY to
   BookerFormer impact_def for that player-window; the gap is allocated in proportion to
   each component's sampling variance (the noisier zone absorbs more of it).

Output cache/def_zones.csv: season, PLAYER_ID, def_interior, def_perimeter, impact_def,
raw_interior, raw_perimeter, def_poss. Run: python -m forecast.def_zones
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from . import player_impacts as pi

CACHE = pi.CACHE
RAPM = CACHE.parent
ALPHA = 2800.0
INTERIOR = {"Restricted Area", "In The Paint (Non-RA)"}


V3_FIX = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}


def _events_classic(s):
    e = pd.read_csv(CACHE / f"nbastats_{s}.csv", low_memory=False,
                    usecols=["GAME_ID", "PERIOD", "EVENTNUM", "PCTIMESTRING", "EVENTMSGTYPE",
                             "PLAYER1_ID", "PLAYER1_TEAM_ID", "HOMEDESCRIPTION", "VISITORDESCRIPTION"])
    e = e[e.EVENTMSGTYPE.isin([2, 3, 4, 5])].copy()
    e["team"] = e.PLAYER1_TEAM_ID.fillna(e.PLAYER1_ID.where(e.PLAYER1_ID >= 1610612737))
    # team rebounds (PLAYER1_ID = team id) are mostly dead-ball entries after a missed
    # non-final free throw -- not real rebounds; keep only player rebounds
    e = e[~((e.EVENTMSGTYPE == 4) & (e.PLAYER1_ID >= 1610612737))]
    desc = (e.HOMEDESCRIPTION.fillna("") + " " + e.VISITORDESCRIPTION.fillna("")).str.upper()
    e["miss"] = (e.EVENTMSGTYPE == 2) | ((e.EVENTMSGTYPE == 3) & desc.str.contains("MISS"))
    mm, ss = e.PCTIMESTRING.str.split(":", expand=True).astype(float).T.values
    e["sec"] = mm * 60 + ss
    return e.rename(columns={"EVENTMSGTYPE": "etype"})[["GAME_ID", "PERIOD", "EVENTNUM", "sec", "etype", "team", "miss"]]


def _events_v3(s):
    import glob
    out = []
    for f in sorted(glob.glob(str(CACHE / f"pbp_{s}" / "*.parquet"))):
        d = pd.read_parquet(f, columns=["gameId", "actionNumber", "clock", "period", "teamId",
                                        "personId", "actionType", "shotResult"])
        d = d[d.actionType.isin(["Missed Shot", "Free Throw", "Rebound", "Turnover"])]
        d = d[~((d.actionType == "Rebound") & (pd.to_numeric(d.personId, errors="coerce").fillna(0) == 0))]
        etype = d.actionType.map({"Missed Shot": 2, "Free Throw": 3, "Rebound": 4, "Turnover": 5})
        clk = d.clock.str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
        out.append(pd.DataFrame({"GAME_ID": d.gameId.astype("int64"), "PERIOD": d.period.astype(int),
                                 "EVENTNUM": d.actionNumber.astype(int), "sec": clk[0] * 60 + clk[1].round(),
                                 "etype": etype, "team": pd.to_numeric(d.teamId, errors="coerce"),
                                 "miss": (d.actionType == "Missed Shot") | ((d.actionType == "Free Throw") & (d.shotResult == "Missed"))}))
    return pd.concat(out, ignore_index=True)


def stint_events(s, st, ab, home):
    """Per stint & offense side: turnovers committed and offensive rebounds."""
    e = _events_classic(s) if (CACHE / f"nbastats_{s}.csv").exists() else _events_v3(s)
    e = e.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"])
    e["miss_team"] = e.team.where(e.miss).groupby([e.GAME_ID, e.PERIOD]).ffill()
    e["oreb"] = (e.etype == 4) & (e.team == e.miss_team)
    e["tov"] = e.etype == 5
    e = e[e.oreb | e.tov].copy()
    e["off_home"] = [ab.get(int(t)) == home.get(int(g)) if t == t else False for t, g in zip(e.team, e.GAME_ID)]
    m = pd.merge_asof(e.sort_values("sec"), st[["GAME_ID", "PERIOD", "START_SEC", "sid"]].sort_values("START_SEC"),
                      by=["GAME_ID", "PERIOD"], left_on="sec", right_on="START_SEC", direction="forward",
                      allow_exact_matches=False).dropna(subset=["sid"])   # event AT a sub clock -> outgoing lineup
    for side, h in (("home", True), ("away", False)):
        mm = m[m.off_home == h]
        st[f"{side}_tov"] = mm[mm.tov].groupby("sid").size().reindex(st.sid, fill_value=0).to_numpy(float)
        st[f"{side}_oreb"] = mm[mm.oreb].groupby("sid").size().reindex(st.sid, fill_value=0).to_numpy(float)
    return st


def stint_zone_points(s):
    """Stints of season s with opponent points split by zone, for both sides."""
    st = pd.read_csv(CACHE / f"stints_{s}.csv")
    st = st[st.GAME_ID.astype(str).str.startswith("2")].reset_index(drop=True)
    if "START_SEC" not in st.columns:              # v3-built seasons: contiguous clocks
        plen = np.where(st.PERIOD <= 4, 720.0, 300.0)
        el = st.groupby(["GAME_ID", "PERIOD"]).DURATION_SECONDS.cumsum() - st.DURATION_SECONDS
        st["START_SEC"] = np.maximum(plen - el, 1.0)
    st["sid"] = np.arange(len(st))
    sh = pd.read_csv(CACHE / f"shotdetail_{s}.csv", low_memory=False,
                     usecols=["GAME_ID", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "TEAM_ID",
                              "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_MADE_FLAG"])
    sh["sec"] = sh.MINUTES_REMAINING * 60.0 + sh.SECONDS_REMAINING
    sh = sh.sort_values("sec")
    m = pd.merge_asof(sh, st[["GAME_ID", "PERIOD", "START_SEC", "sid"]].sort_values("START_SEC"),
                      by=["GAME_ID", "PERIOD"], left_on="sec", right_on="START_SEC", direction="forward",
                      allow_exact_matches=False)   # event AT a sub clock belongs to the outgoing lineup
    m = m.dropna(subset=["sid"])
    teams = pd.read_csv(CACHE / f"teams_{s}.csv")
    ab = dict(zip(teams.TEAM_ID.astype(int), teams.ABBR))
    g = pd.read_csv(CACHE / f"games_{s}.csv")
    home = dict(zip(g.GAME_ID.astype(int), g.HOME))
    m["shooter_home"] = [ab.get(int(t)) == home.get(int(gid)) for t, gid in zip(m.TEAM_ID, m.GAME_ID)]
    m["pts"] = m.SHOT_MADE_FLAG * np.where(m.SHOT_TYPE.str.startswith("3"), 3, 2)
    m["inside"] = m.SHOT_ZONE_BASIC.isin(INTERIOR)
    agg = m.groupby(["sid", "shooter_home", "inside"]).pts.agg(["sum", "size"]).unstack(["shooter_home", "inside"], fill_value=0)
    def col(stat, h, i):
        k = (stat, h, i)
        return agg[k].reindex(st.sid, fill_value=0).to_numpy(float) if k in agg.columns else np.zeros(len(st))
    for side, h in (("home", True), ("away", False)):
        st[f"{side}_in_fg"], st[f"{side}_out_fg"] = col("sum", h, True), col("sum", h, False)
        st[f"{side}_in_n"], st[f"{side}_out_n"] = col("size", h, True), col("size", h, False)
    # free throws = side points not explained by mapped FG points (clipped at 0)
    st["home_ft"] = np.maximum(st.HOME_PTS - st.home_in_fg - st.home_out_fg, 0)
    st["away_ft"] = np.maximum(st.AWAY_PTS - st.away_in_fg - st.away_out_fg, 0)
    st["home"] = st.HOME_LINEUP.map(pi.parse_lineup); st["away"] = st.AWAY_LINEUP.map(pi.parse_lineup)
    ab2 = dict(ab)
    for tid, a in list(ab.items()):
        ab2[tid] = a
    return stint_events(s, st, {**ab2, **{k: V3_FIX.get(v, v) for k, v in ab2.items()}}, home)


def components(d, side):
    """Exact decomposition of one side's offense in each stint (points, league-relative)."""
    pin, pout = d[f"{side}_in_fg"], d[f"{side}_out_fg"]
    nin, nout = d[f"{side}_in_n"], d[f"{side}_out_n"]
    ft = d[f"{side}_ft"]
    return pin, pout, nin, nout, ft


def league_rates(d):
    n_in = d.home_in_n.sum() + d.away_in_n.sum(); n_out = d.home_out_n.sum() + d.away_out_n.sum()
    e_in = (d.home_in_fg.sum() + d.away_in_fg.sum()) / n_in
    e_out = (d.home_out_fg.sum() + d.away_out_fg.sum()) / n_out
    sh = n_in / (n_in + n_out)
    poss = 2 * d.POSS.sum()
    return dict(e_in=e_in, e_out=e_out, sh=sh, e_bar=sh * e_in + (1 - sh) * e_out,
                fga_rate=(n_in + n_out) / poss, ft_rate=(d.home_ft.sum() + d.away_ft.sum()) / poss,
                tov_rate=(d.home_tov.sum() + d.away_tov.sum()) / poss,
                oreb_rate=(d.home_oreb.sum() + d.away_oreb.sum()) / poss)


# interior  = rim/paint efficiency + deterrence + free throws + offensive rebounds allowed
# perimeter = jumper/three efficiency + turnovers forced
# pace      = residual shot volume not explained by turnovers / offensive rebounds
COMPS = ("interior", "perimeter", "pace")


def fit_window(stints, decay_target, rows_mask=None):
    """Ridge per component over a multi-season window. Returns {comp: {pid: d}}, poss."""
    ids = sorted({p for d in stints.values() for l in (d.home, d.away) for x in l for p in x})
    col = {p: i for i, p in enumerate(ids)}
    P = len(ids)
    r, c, v, w = [], [], [], []
    Y = {k: [] for k in COMPS}
    poss = {}
    k = 0
    for s, d in stints.items():
        L = league_rates(d)
        dw = pi.DECAY ** (decay_target - 1 - s)
        if rows_mask is not None:
            d = d[rows_mask(d)]
        for side_off, side_def in (("home", "away"), ("away", "home")):
            pin, pout, nin, nout, ft = components(d, side_off)
            N = nin + nout
            per100 = 100.0 / d.POSS
            tov = (d[f"{side_off}_tov"] - L["tov_rate"] * d.POSS) * L["e_bar"]      # extra TOs -> fewer shots
            oreb = (d[f"{side_off}_oreb"] - L["oreb_rate"] * d.POSS) * L["e_bar"]   # extra OREBs -> more shots
            vol = (N - L["fga_rate"] * d.POSS) * L["e_bar"]
            yi = ((pin - nin * L["e_in"]) + (nin - L["sh"] * N) * (L["e_in"] - L["e_out"])
                  + (ft - L["ft_rate"] * d.POSS) + oreb) * per100
            yp = ((pout - nout * L["e_out"]) - tov) * per100
            yv = (vol - oreb + tov) * per100
            for off, de, a, b, cc, ps in zip(d[side_off], d[side_def], yi, yp, yv, d.POSS):
                for p in off:
                    r.append(k); c.append(col[p]); v.append(1.0)
                for p in de:
                    r.append(k); c.append(P + col[p]); v.append(-1.0)
                    poss[p] = poss.get(p, 0.0) + ps
                Y["interior"].append(a); Y["perimeter"].append(b); Y["pace"].append(cc)
                w.append(ps * dw); k += 1
    X = csr_matrix((v, (r, c)), shape=(k, 2 * P))
    w = np.array(w); out = {}
    for comp in COMPS:
        y = np.array(Y[comp])
        m = Ridge(alpha=ALPHA).fit(X, y, sample_weight=w)
        out[comp] = dict(zip(ids, m.coef_[P:]))            # defense block: + = suppresses
        out[comp + "_var"] = float(np.average((y - m.predict(X)) ** 2, weights=w))
    return out, poss


def build(seasons=range(2017, 2027)):
    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    cache = {}
    rows = []
    for s in seasons:
        win = {t: cache.setdefault(t, stint_zone_points(t)) for t in range(s - 2, s + 1)
               if (CACHE / f"shotdetail_{t}.csv").exists() and (CACHE / f"stints_{t}.csv").exists()}
        if len(win) < 2:
            continue
        comp, poss = fit_window(win, s + 1)
        var = {k: comp[k + "_var"] for k in COMPS}
        vt = sum(var.values())
        r = rat[rat.season == s]
        for pid, d_bf in zip(r.PLAYER_ID, r.impact_def):
            raw = {k: comp[k].get(pid, 0.0) for k in COMPS}
            gap = d_bf - sum(raw.values())
            row = {"season": s, "PLAYER_ID": int(pid), "impact_def": round(d_bf, 2),
                   "def_poss": round(poss.get(pid, 0.0))}
            for k in COMPS:
                row[f"def_{k}"] = round(raw[k] + gap * var[k] / vt, 2)
                row[f"raw_{k}"] = round(raw[k], 2)
            rows.append(row)
        vi, vp = var["interior"], var["perimeter"]
        print(f"  {s}: {len(r)} players (zone noise var interior {vi:.0f}, perimeter {vp:.0f})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "def_zones.csv", index=False)
    print(f"wrote {CACHE / 'def_zones.csv'} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    build()
