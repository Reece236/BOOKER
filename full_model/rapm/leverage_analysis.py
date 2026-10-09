"""Does leverage change how plays should be valued -- and does creation hold up?

Panel: every regular-season 5v5 stint 2018-2026, two offense rows per stint (home O
vs away D, away O vs home D), real points, leverage from forecast.leverage.
Residual r = offense pts/100 - season league - (sum o_off - sum d_def) using
BookerFormer O/D ratings THROUGH that season (leverage-agnostic averages).

Tests
  1. PREMISE   residual offense by LI bucket (does defense tighten / offense degrade?)
  2. MECHANISM WLS r ~ LIc + LIc x creator_z + LIc x offQuality_z + LIc x oppDef_z
               + late-game lead/trail controls (intentional fouling), poss-weighted,
               game-clustered SEs. LIc = LI - 1.
  3. PLAYERS   per player-season high-leverage offensive residual (LI>2) minus his
               normal residual; year-over-year reliability; named players.
  4. VALUE     deployment leverage (poss-weighted mean LI of a player's minutes) and
               how much LI-weighting moves value.
Run: python leverage_analysis.py
"""
import numpy as np
import pandas as pd

from forecast import player_impacts as pi
from forecast import bookerformer as bf

CACHE = pi.CACHE
SEASONS = range(2018, 2027)
NAMED = {1628973: "Jalen Brunson", 1627759: "Jaylen Brown", 1628983: "Shai Gilgeous-Alexander",
         203999: "Nikola Jokic", 1628369: "Jayson Tatum", 1629029: "Luka Doncic",
         202710: "Jimmy Butler", 1626164: "Devin Booker", 203507: "Giannis Antetokounmpo",
         1628378: "Donovan Mitchell", 201939: "Stephen Curry", 1630178: "Tyrese Maxey"}


def wls_cluster(X, y, w, groups):
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    beta = np.linalg.solve(XtWX, Xw.T @ y)
    e = y - X @ beta
    inv = np.linalg.inv(XtWX)
    s = pd.DataFrame(Xw * e[:, None]).groupby(groups).sum().to_numpy()
    V = inv @ (s.T @ s) @ inv
    return beta, np.sqrt(np.diag(V))


def panel():
    rat = pd.read_csv("booker_bookerformer_ratings.csv")
    sq = pd.read_csv(CACHE / "shot_quality.csv")
    L = pd.read_parquet(CACHE / "stint_leverage.parquet")
    rows = []
    for s in SEASONS:
        r = rat[rat.season == s]
        o = dict(zip(r.PLAYER_ID, r.impact_off)); d = dict(zip(r.PLAYER_ID, r.impact_def))
        mins = dict(zip(r.PLAYER_ID, r.minutes))
        q = sq[(sq.season == s)]
        q = q[q.PLAYER_ID.map(mins).fillna(0) >= 500]
        cz = dict(zip(q.PLAYER_ID, (q.self_create - q.self_create.mean()) / q.self_create.std()))
        st = pd.read_csv(CACHE / f"stints_{s}.csv")
        st = st[st.GAME_ID.astype(str).str.startswith("2")].reset_index(drop=True)
        st["row"] = np.arange(len(st))
        st = st.merge(L[L.season == s][["GAME_ID", "row", "LI", "margin", "tau"]], on=["GAME_ID", "row"])
        lg = bf._season_league(st)
        for side in ("home", "away"):
            off = st.HOME_LINEUP if side == "home" else st.AWAY_LINEUP
            de = st.AWAY_LINEUP if side == "home" else st.HOME_LINEUP
            y = st.Y_OFF_HOME if side == "home" else st.Y_DEF_HOME
            lead = st.margin if side == "home" else -st.margin
            for gid, ol, dl, yy, li, ld, tau, poss in zip(st.GAME_ID, off, de, y, st.LI, lead, st.tau, st.POSS):
                op = [int(x) for x in ol.split(",")]; dp = [int(x) for x in dl.split(",")]
                pred = sum(o.get(p, -0.5) for p in op) - sum(d.get(p, -0.5) for p in dp)
                cr = [cz[p] for p in op if p in cz]
                rows.append((s, gid, side == "home", yy - lg - pred, li, ld, tau, poss,
                             max(cr) if cr else np.nan, sum(o.get(p, -0.5) for p in op),
                             sum(d.get(p, -0.5) for p in dp), tuple(op)))
    P = pd.DataFrame(rows, columns=["season", "GAME_ID", "home", "r", "LI", "lead", "tau", "POSS",
                                    "create_max", "off_sum", "def_sum", "offense"])
    return P[P.POSS > 0.5]          # sub-15-second stints are pure noise per 100


def main():
    P = panel()
    print(f"panel: {len(P):,} offense-stints, {P.POSS.sum():,.0f} possessions")
    # ---- 1. premise
    P["bucket"] = pd.cut(P.LI, [-0.01, 0.25, 0.75, 1.25, 2, 4, 99],
                         labels=["<.25 (garbage)", ".25-.75", ".75-1.25", "1.25-2", "2-4", "4+ (crunch)"])
    b = P.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({"share_poss": g.POSS.sum() / P.POSS.sum(),
                             "resid_off_per100": np.average(g.r, weights=g.POSS)}), include_groups=False)
    print("\n1. PREMISE: offense vs. what the 10 players' ratings predict, by leverage")
    print(b.round(3).to_string())
    # ---- 2. mechanism
    D = P.dropna(subset=["create_max"]).copy()
    z = lambda v: (v - v.mean()) / v.std()
    D["LIc"] = D.LI - 1.0
    D["cz"], D["oz"], D["dz"] = z(D.create_max), z(D.off_sum), z(D.def_sum)
    late = D.tau < (3 * 60 / 2880)
    D["late_lead"] = (late & (D.lead > 0)).astype(float)
    D["late_trail"] = (late & (D.lead < 0)).astype(float)
    cols = ["const", "LIc", "LIc_x_creator", "LIc_x_offQual", "LIc_x_oppDef", "creator", "home",
            "late_lead", "late_trail"]
    X = np.column_stack([np.ones(len(D)), D.LIc, D.LIc * D.cz, D.LIc * D.oz, D.LIc * D.dz, D.cz,
                         D.home.astype(float), D.late_lead, D.late_trail])
    beta, se = wls_cluster(X, D.r.to_numpy(), D.POSS.to_numpy(), D.GAME_ID.to_numpy())
    print("\n2. MECHANISM (pts/100; LIc = LI - 1; z = per SD; game-clustered SE)")
    for c, b_, s_ in zip(cols, beta, se):
        print(f"  {c:15s} {b_:+7.3f}  (se {s_:.3f}, t {b_ / s_:+.1f})")
    for s0 in (2018, 2022):           # stability: two halves
        m = (D.season >= s0) & (D.season < s0 + 4 + (s0 == 2022))
        bb, ss = wls_cluster(X[m.to_numpy()], D.r[m].to_numpy(), D.POSS[m].to_numpy(), D.GAME_ID[m].to_numpy())
        print(f"  [{s0}-{s0+3+(s0==2022)}] LIc_x_creator {bb[2]:+.3f} (t {bb[2]/ss[2]:+.1f}), LIc_x_offQual {bb[3]:+.3f} (t {bb[3]/ss[3]:+.1f})")
    # ---- 3. players
    ex = P[["season", "offense", "r", "LI", "POSS"]].explode("offense").rename(columns={"offense": "PLAYER_ID"})
    ex["hi"] = ex.LI > 2
    g = ex.groupby(["season", "PLAYER_ID", "hi"]).apply(
        lambda x: pd.Series({"r": np.average(x.r, weights=x.POSS), "poss": x.POSS.sum()}), include_groups=False).unstack("hi")
    g.columns = [f"{a}_{'hi' if b else 'lo'}" for a, b in g.columns]
    g = g.dropna()
    g = g[g.poss_hi >= 150]
    g["diff"] = g.r_hi - g.r_lo
    g = g.reset_index()
    nxt = g[["season", "PLAYER_ID", "diff"]].copy(); nxt["season"] -= 1
    yy = g.merge(nxt, on=["season", "PLAYER_ID"], suffixes=("", "_next"))
    print(f"\n3. PLAYERS: high-leverage (LI>2) minus normal offensive residual, >=150 hi-LI poss")
    print(f"  year-over-year reliability r = {np.corrcoef(yy['diff'], yy.diff_next)[0,1]:+.3f} (n={len(yy)} pairs)")
    car = g.groupby("PLAYER_ID").apply(lambda x: pd.Series({
        "hi_poss": x.poss_hi.sum(), "diff": np.average(x["diff"], weights=x.poss_hi),
        "se": 160 / np.sqrt(x.poss_hi.sum())}), include_groups=False)
    for pid, nm in NAMED.items():
        if pid in car.index:
            c = car.loc[pid]
            print(f"  {nm:24s} career hi-LI diff {c['diff']:+6.1f} pts/100  +- {1.96*c.se:4.1f}  ({c.hi_poss:,.0f} hi-LI poss)")
    # ---- 4. value
    dep = ex.groupby(["season", "PLAYER_ID"]).apply(
        lambda x: pd.Series({"poss": x.POSS.sum(), "LI_mean": np.average(x.LI, weights=x.POSS)}),
        include_groups=False).reset_index()
    rat = pd.read_csv("booker_bookerformer_ratings.csv")
    dep = dep.merge(rat[["season", "PLAYER_ID", "player", "impact_total", "minutes"]], on=["season", "PLAYER_ID"])
    d26 = dep[(dep.season == 2026) & (dep.minutes >= 1500)]
    print(f"\n4. VALUE: deployment leverage, 2026 (>=1500 min, n={len(d26)}): "
          f"p10 {d26.LI_mean.quantile(.1):.2f}, median {d26.LI_mean.median():.2f}, p90 {d26.LI_mean.quantile(.9):.2f}")
    print("  corr(deployment LI, impact) = %+.2f" % np.corrcoef(d26.LI_mean, d26.impact_total)[0, 1])
    print(d26[d26.PLAYER_ID.isin(NAMED)].sort_values("LI_mean", ascending=False)[["player", "LI_mean", "impact_total"]].round(2).to_string(index=False))
    P.drop(columns=["offense"]).to_parquet(CACHE / "leverage_panel.parquet")
    dep.to_csv(CACHE / "leverage_deployment.csv", index=False)


if __name__ == "__main__":
    main()
