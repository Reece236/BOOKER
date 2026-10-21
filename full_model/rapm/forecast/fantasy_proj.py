"""
Fantasy (H2H 8-cat) projections: per-game PTS, REB, AST, STL, BLK, 3PM, FGM/FGA, FTM/FTA.

Rates (per minute) are Marcel-style: seasons s-1, s-2, s-3 weighted 5/4/3 by minutes, plus
K_stat minutes of the player's position-group league rate (regression to the mean; noisier
stats get bigger K). Percentages are projected separately: FG% / FT% with K_attempts of the
league rate. An age term per stat (fit on prior seasons) scales the rates. K's are chosen
walk-forward (tune 2019-23 targets, report 2024-26) against the naive last-season rate.

Projected season minutes come from forecast.minutes_share (cache/minutes_proj.csv);
minutes per game = Marcel of recent minutes per game; games = minutes / mpg, capped at 82.

  python -m forecast.fantasy_proj           # validate + write cache/fantasy_proj_2027.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import player_impacts as pi

CACHE = pi.CACHE
OUT = CACHE / "fantasy_proj_2027.csv"
COUNT = {"pts": "total_points", "reb": "total_totalRb", "ast": "total_assists", "stl": "total_steals",
         "blk": "total_blocks", "tpm": "total_threeFg", "fga": "total_fieldAttempts", "fta": "total_ftAttempts",
         "tov": "total_turnovers"}
MAKES = {"fg": ("total_fieldGoals", "total_fieldAttempts"), "ft": ("total_ft", "total_ftAttempts")}
POSG = {"PG": "G", "SG": "G", "SF": "W", "PF": "B", "C": "B"}
W = (5.0, 4.0, 3.0)
ROOKIE_MULT = 0.85
W_CUR = 6.0            # live season-to-date weight per minute (vs 5/4/3 for the prior seasons)     # first-year counting rates vs the position-group league rate
K_GRID = (150, 300, 600, 1200, 2400)
KP_GRID = (50, 150, 400, 1000)


def _box():
    b = pi.load_box()
    b = b[b.minutesPlayed > 0].copy()
    b["nm"] = b.playerName.map(pi.norm_name)
    b["pg"] = b.position.astype(str).str.split("-").str[0].map(POSG).fillna("W")
    return b


def _league(b, s):
    x = b[(b.season == s) & (b.minutesPlayed >= 500)]
    lr = {}
    for g, d in x.groupby("pg"):
        m = d.minutesPlayed.sum()
        lr[g] = {k: d[c].sum() / m for k, c in COUNT.items()}
        for k, (mk, at) in MAKES.items():
            lr[g][k + "_pct"] = d[mk].sum() / d[at].sum()
    return lr


def project(b, target, K, KP, age_c=None, cur=False):
    """Per-minute rates + pcts for every player with any of the 3 prior seasons. cur=True
    (LIVE season only, never in validation) also uses the target season's games to date,
    weighted W_CUR per minute -- rest-of-season rates track the season as it unfolds."""
    lr = _league(b, target - 1)
    prior = b[b.season.between(target - 3, target if cur else target - 1)]
    rows = []
    for nm, d in prior.groupby("nm"):
        wts = {target - 1 - i: W[i] for i in range(3)}
        wts[target] = W_CUR
        d = d.assign(w=d.season.map(wts))
        wm = (d.w * d.minutesPlayed).sum()
        last = d[d.season < target].sort_values("season").iloc[-1] if (d.season < target).any() else d.iloc[-1]
        g = last.pg
        r = {"nm": nm, "player": last.playerName, "pos": last.position, "pg": g,
             "age": float(last.age) + (target - int(last.season)), "wmin": wm}
        for k, c in COUNT.items():
            r[k] = ((d.w * d[c]).sum() + K[k] * lr[g][k]) / (wm + K[k])
        for k, (mk, at) in MAKES.items():
            att = (d.w * d[at]).sum()
            r[k + "_pct"] = ((d.w * d[mk]).sum() + KP[k] * lr[g][k + "_pct"]) / (att + KP[k])
        gw = (d.w * d.games).sum()
        r["mpg"] = ((d.w * d.minutesPlayed).sum() + 5.0 * 22.0) / (gw + 5.0)   # + 5 games at 22 mpg
        rows.append(r)
    P = pd.DataFrame(rows)
    if age_c:
        for k in COUNT:
            P[k] = P[k] * np.clip(1 + age_c.get(k, 0.0) * (P.age - 27.0), 0.7, 1.3)
    return P


def _actual(b, s):
    a = b[(b.season == s) & (b.minutesPlayed >= 1000)].copy()
    out = pd.DataFrame({"nm": a.nm, "min": a.minutesPlayed})
    for k, c in COUNT.items():
        out[k] = (a[c] / a.minutesPlayed).values
    for k, (mk, at) in MAKES.items():
        out[k + "_pct"] = (a[mk] / a[at].clip(lower=1)).values
        out[k + "_att"] = a[at].values
    return out


def _err(P, A, k):
    m = A.merge(P[["nm", k]], on="nm", suffixes=("", "_p"))
    if k.endswith("_pct"):
        w = m[k.replace("_pct", "_att")]
        return float(np.sqrt(np.average((m[k] - m[k + "_p"]) ** 2, weights=w))), len(m)
    return float(np.sqrt(np.average((36 * (m[k] - m[k + "_p"])) ** 2, weights=m["min"]))), len(m)


def fit_age(b, targets, K, KP):
    """Per-stat linear age multiplier on the Marcel rate, fit on `targets` (prior seasons)."""
    xs, ys = {k: [] for k in COUNT}, {k: [] for k in COUNT}
    for t in targets:
        P, A = project(b, t, K, KP), _actual(b, t)
        m = A.merge(P, on="nm", suffixes=("", "_p"))
        for k in COUNT:
            ok = m[k + "_p"] > 0
            xs[k] += list(m.age[ok] - 27.0); ys[k] += list((m[k][ok] / m[k + "_p"][ok] - 1.0).clip(-1, 1))
    return {k: float(np.polyfit(xs[k], ys[k], 1)[0]) if len(xs[k]) > 50 else 0.0 for k in COUNT}


def validate(b):
    tune, test = range(2019, 2024), range(2024, 2027)
    K, KP = {}, {}
    for k in COUNT:
        K[k] = min(K_GRID, key=lambda kk: np.mean([_err(project(b, t, {**{c: 600 for c in COUNT}, k: kk},
                                                                {"fg": 150, "ft": 150}), _actual(b, t), k)[0] for t in tune]))
    for k in MAKES:
        KP[k] = min(KP_GRID, key=lambda kk: np.mean([_err(project(b, t, K, {**{"fg": 150, "ft": 150}, k: kk}),
                                                          _actual(b, t), k + "_pct")[0] for t in tune]))
    print("regression K (minutes):", K, "| pct K (attempts):", KP)
    rows = []
    for t in test:
        age_c = fit_age(b, range(2019, t), K, KP)
        P, A = project(b, t, K, KP, age_c), _actual(b, t)
        naive = b[b.season == t - 1].copy()
        N = pd.DataFrame({"nm": naive.nm})
        for k, c in COUNT.items():
            N[k] = (naive[c] / naive.minutesPlayed).values
        for k, (mk, at) in MAKES.items():
            N[k + "_pct"] = (naive[mk] / naive[at].clip(lower=1)).values
        for k in list(COUNT) + ["fg_pct", "ft_pct"]:
            common = set(P.nm) & set(N.nm)
            e1, n = _err(P[P.nm.isin(common)], A, k); e0, _ = _err(N[N.nm.isin(common)], A, k)
            rows.append({"target": t, "stat": k, "marcel": e1, "naive": e0, "n": n})
    R = pd.DataFrame(rows).groupby("stat")[["marcel", "naive"]].mean()
    R["gain_%"] = (100 * (1 - R.marcel / R.naive)).round(1)
    print("walk-forward test 2024-26 (per-36 RMSE; pct in pct points x100):")
    print(R.round(4).to_string())
    return K, KP


def _current_box(season, b):
    """ESPN season-to-date totals (data_ingest/fetch_espn_season.py) in the bbref schema."""
    p = CACHE / f"espn_box_{season}.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p)
    e = e[e["min"] > 0]
    if e.empty:
        return None
    g = e.groupby("NAME").agg(games=("GAME_ID", "nunique"), minutesPlayed=("min", "sum"), total_points=("pts", "sum"),
                              total_totalRb=("reb", "sum"), total_assists=("ast", "sum"), total_steals=("stl", "sum"),
                              total_blocks=("blk", "sum"), total_threeFg=("tpm", "sum"), total_fieldAttempts=("fga", "sum"),
                              total_fieldGoals=("fgm", "sum"), total_ftAttempts=("fta", "sum"), total_ft=("ftm", "sum"),
                              total_turnovers=("tov", "sum")).reset_index().rename(columns={"NAME": "playerName"})
    g["nm"] = g.playerName.map(pi.norm_name)
    last = b.sort_values("season").drop_duplicates("nm", keep="last").set_index("nm")
    g["position"] = g.nm.map(last.position).fillna("SF")
    g["pg"] = g.nm.map(last.pg).fillna("W")
    g["age"] = g.nm.map(last.age + (season - last.season)).fillna(21.0)
    g["season"] = season
    return g


def build(write=True):
    b = _box()
    K, KP = validate(b)
    age_c = fit_age(b, range(2019, 2027), K, KP)
    print("age multiplier per year (x(age-27)):", {k: round(v, 4) for k, v in age_c.items()})
    cur = _current_box(2027, b)
    if cur is not None:
        b = pd.concat([b, cur], ignore_index=True)
        print(f"live 2027: season-to-date box for {len(cur)} players folded in (weight {W_CUR}/min)")
    P = project(b, 2027, K, KP, age_c, cur=cur is not None)
    m = pd.read_csv(CACHE / "minutes_proj.csv")
    m = m[m.season == 2027].copy()
    m["nm"] = m.name.map(pi.norm_name)
    # name bridge: Jr/II suffixes and nicknames ("Ronald Holland II" ~ "Ron Holland")
    SUF = {"jr", "sr", "ii", "iii", "iv", "v"}
    toks = lambda n: [w for w in n.replace(".", "").split() if w not in SUF]
    by_last = {}
    for nm in P.nm:
        tk = toks(nm)
        if len(tk) >= 2:
            by_last.setdefault(tk[-1], []).append((tk[0], nm))
    def bridge(nm):
        if nm in set(P.nm):
            return nm
        tk = toks(nm)
        if len(tk) < 2:
            return nm
        c = [q for f, q in by_last.get(tk[-1], []) if f.startswith(tk[0]) or tk[0].startswith(f)]
        return c[0] if len(c) == 1 else nm
    m["nm"] = m.nm.map(bridge)
    out = m.merge(P, on="nm", how="left")
    # ROOKIES / no NBA box history: position-group average rookie line (first-year players
    # produce below league rate); flagged so the UI can mark them
    rk = out.player.isna()
    lr = _league(b, 2026)
    mast = b.sort_values("season").drop_duplicates("nm", keep="last").set_index("nm").position
    for i in out.index[rk]:
        g = POSG.get(str(mast.get(out.at[i, "nm"], "SF")).split("-")[0], "W")
        out.at[i, "player"] = out.at[i, "name"]; out.at[i, "pos"] = mast.get(out.at[i, "nm"], "-")
        out.at[i, "pg"] = g; out.at[i, "age"] = 21.0; out.at[i, "mpg"] = 18.0
        for k in COUNT:
            out.at[i, k] = lr[g][k] * ROOKIE_MULT
        out.at[i, "fg_pct"] = lr[g]["fg_pct"] - 0.02; out.at[i, "ft_pct"] = lr[g]["ft_pct"] - 0.03
    out["rookie"] = rk.astype(int)
    out["mpg"] = out.mpg.clip(8, 38)
    out["games"] = (out.proj_minutes / out.mpg).clip(0, 82).round(1)
    # per-game line at projected minutes per game
    for k in COUNT:
        out[k + "_g"] = (out[k] * out.mpg).round(3)
    out["fgm_g"] = (out.fga_g * out.fg_pct).round(3)
    out["ftm_g"] = (out.fta_g * out.ft_pct).round(3)
    cols = ["PLAYER_ID", "player", "team", "pos", "age", "games", "mpg", "proj_minutes",
            "pts_g", "reb_g", "ast_g", "stl_g", "blk_g", "tpm_g", "fgm_g", "fga_g", "ftm_g", "fta_g", "tov_g",
            "fg_pct", "ft_pct", "rookie"]
    out = out[cols].sort_values("proj_minutes", ascending=False)
    if write:
        out.to_csv(OUT, index=False)
        print(f"wrote {OUT} ({len(out)} players; {int((out.games >= 20).sum())} with 20+ projected games)")
    return out


if __name__ == "__main__":
    build()
