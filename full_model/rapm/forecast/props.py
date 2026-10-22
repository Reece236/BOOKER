"""
Game-level player prop projections: PTS, REB, AST, 3PM, STL, BLK, TOV, PRA.

For player p in game g (only information before g's date):
  rate (per possession) = (r0 * Kp + counts to date) / (Kp + poss to date)
      r0 = preseason Marcel rate (forecast.fantasy_proj.project, per minute) converted to
           per possession; Kp = that stat's tuned regression minutes x possessions/minute
  poss  = blend(projected mpg x poss/min, season-to-date poss per game played; 5-game prior)
          x game pace factor ((team pace + opponent pace) / 2 / team pace; to-date, prior-blended)
  mu    = rate x poss;  y ~ NegBin(mean mu, size k_stat)  (var = mu + mu^2 / k)
Props are conditional on playing (books void DNPs), so only games played are scored.

Backtest 2025-26 (python -m forecast.props --backtest): fit k on games before the season
midpoint, score the second half: MAE vs (a) last season's per-game average and (b) the
season-to-date average; calibration of P(over line) at the lines a book would hang
(projection rounded to .5) and 80% interval coverage.

Live (default): next 7 days of the in-progress season -> cache/props_live.csv.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.special import gammaln

from . import player_impacts as pi
from . import fantasy_proj as fp

CACHE = pi.CACHE
STATS = ["pts", "reb", "ast", "tpm", "stl", "blk", "tov"]
KMIN = {"pts": 1200, "reb": 600, "ast": 600, "tpm": 600, "stl": 2400, "blk": 2400, "tov": 1200}
G_PRIOR = 5.0               # games of prior weight on possessions per game
K_MULT = 1.0                # in-season prior weight multiplier on KMIN (tuned in backtest)
ERA = True                  # scale prior rates by the league's season-to-date / prior-season rate
PACE_PRIOR = 10.0           # team-games of prior weight on pace


def nb_logpmf(y, mu, k):
    mu = np.maximum(mu, 1e-6)
    return (gammaln(y + k) - gammaln(k) - gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))


def nb_cdf(y, mu, k):
    """P(Y <= y) for integer arrays y (vectorised by summing the pmf)."""
    ymax = int(np.max(y)) if np.size(y) else 0
    grid = np.arange(ymax + 1)
    pm = np.exp(nb_logpmf(grid[None, :], np.asarray(mu)[:, None], k))
    cs = np.cumsum(pm, axis=1)
    return cs[np.arange(len(y)), np.clip(y, 0, ymax)]


def _priors(season):
    b = fp._box()
    K, KP = {k: KMIN.get(k, 600) for k in fp.COUNT}, {"fg": 1000, "ft": 400}
    age_c = fp.fit_age(b, range(2019, season), K, KP)
    P = fp.project(b, season, K, KP, age_c)
    return P


def _pace_prior(data, season):
    """Possessions per 48 team-minutes by team, prior season (stints)."""
    st = data.STINTS[season - 1]
    g = data.GAMES[season - 1]
    home = dict(zip(g.GAME_ID.astype("int64"), g.HOME)); away = dict(zip(g.GAME_ID.astype("int64"), g.AWAY))
    acc = {}
    for gid, poss, dur in zip(st.GAME_ID.astype("int64"), st.POSS, st.DURATION_SECONDS):
        for t in (home.get(gid), away.get(gid)):
            if t:
                a = acc.setdefault(t, [0.0, 0.0]); a[0] += poss; a[1] += dur
    return {t: 48 * 60 * p / d for t, (p, d) in acc.items() if d > 0}


def season_games(data, season, box):
    """box: per player-game counts with GAME_ID, PLAYER_ID, poss + STATS. Adds date/team/opp."""
    g = data.GAMES[season]
    g = g[g.SEASON_TYPE == "Regular Season"]
    meta = {int(r.GAME_ID): (r.DATE, r.HOME, r.AWAY) for r in g.itertuples()}
    st = data.STINTS[season]
    side = {}
    for gid, hl, al in zip(st.GAME_ID.astype("int64"), st.home, st.away):
        for p in hl:
            side[(gid, p)] = "h"
        for p in al:
            side[(gid, p)] = "a"
    box = box[box.GAME_ID.isin(meta)].copy()
    box["date"] = box.GAME_ID.map(lambda x: meta[x][0])
    sd = [side.get((int(a), int(b))) for a, b in zip(box.GAME_ID, box.PLAYER_ID)]
    box["team"] = [meta[int(gid)][1] if s_ == "h" else meta[int(gid)][2] if s_ == "a" else None for gid, s_ in zip(box.GAME_ID, sd)]
    box["opp"] = [meta[int(gid)][2] if s_ == "h" else meta[int(gid)][1] if s_ == "a" else None for gid, s_ in zip(box.GAME_ID, sd)]
    return box.dropna(subset=["team"])


def project_games(data, season, box_to_date, targets, P, ppm, k_mult=None, g_prior=None, era=None, lg_prev=None):
    """mu per (target player-game). box_to_date: all player-games of the season so far (any
    date); for each target only rows with date < target date are used."""
    k_mult = K_MULT if k_mult is None else k_mult
    g_prior = G_PRIOR if g_prior is None else g_prior
    era = ERA if era is None else era
    pr = P.set_index("nm")
    # league per-possession rates to date (cumulative by date) for the era adjustment
    Bd = box_to_date.groupby("date")[STATS + ["poss"]].sum().sort_index().cumsum()
    names = {}
    for s_ in (season, season - 1, season - 2):
        pl = data.PLAYERS.get(s_)
        if pl is not None:
            for pid, nm in zip(pl.PLAYER_ID.astype(int), pl.NAME):
                names.setdefault(pid, pi.norm_name(nm))
    pace0 = _pace_prior(data, season)
    lg_pace = float(np.mean(list(pace0.values())))
    # team pace to date (poss per game per team, from the box poss sums / 5 players)
    tg = box_to_date.groupby(["date", "team", "GAME_ID"]).poss.sum().div(5).reset_index().sort_values("date")
    B = box_to_date.sort_values("date")
    out = []
    by_p = {p: g for p, g in B.groupby("PLAYER_ID")}
    by_t = {t: g for t, g in tg.groupby("team")}
    for r in targets.itertuples():
        nm = names.get(int(r.PLAYER_ID))
        if nm is None or nm not in pr.index:
            continue
        q = pr.loc[nm]
        if isinstance(q, pd.DataFrame):
            q = q.iloc[0]
        hist = by_p.get(r.PLAYER_ID)
        hist = hist[hist.date < r.date] if hist is not None else None
        n = 0 if hist is None else len(hist)
        P_to = 0.0 if n == 0 else float(hist.poss.sum())
        poss_prior = float(q.mpg) * ppm
        poss_g = (g_prior * poss_prior + P_to) / (g_prior + n)
        def tpace(t):
            h = by_t.get(t)
            h = h[h.date < r.date] if h is not None else None
            m = 0 if h is None else len(h)
            base = pace0.get(t, lg_pace)
            return (PACE_PRIOR * base + (0 if m == 0 else h.poss.sum() * 48 / 48)) / (PACE_PRIOR + m) if m else base
        tp, op = tpace(r.team), tpace(r.opp)
        poss_g *= ((tp + op) / 2.0) / tp
        row = {"GAME_ID": r.GAME_ID, "PLAYER_ID": r.PLAYER_ID, "date": r.date, "team": r.team, "opp": r.opp,
               "poss": poss_g, "min": poss_g / ppm}
        prev_d = Bd.index[Bd.index < r.date]
        for s in STATS:
            r0 = float(q[s]) / ppm                       # per possession
            if era and lg_prev and len(prev_d) and Bd.loc[prev_d[-1], "poss"] > 2000:
                lg_now = Bd.loc[prev_d[-1], s] / Bd.loc[prev_d[-1], "poss"]
                w_ = min(1.0, Bd.loc[prev_d[-1], "poss"] / 20000.0)      # trust the league rate as it accrues
                r0 *= (1 - w_) + w_ * lg_now / lg_prev[s]
            kp = KMIN[s] * ppm * k_mult
            C = 0.0 if n == 0 else float(hist[s].sum())
            row[s] = (r0 * kp + C) / (kp + P_to) * poss_g
        row["pra"] = row["pts"] + row["reb"] + row["ast"]
        out.append(row)
    return pd.DataFrame(out)


def _ppm(data, season):
    st = data.STINTS[season - 1]
    return float(st.POSS.sum() / (st.DURATION_SECONDS.sum() / 60.0) / 1.0)   # team poss per minute (= player)


def _league_prev(season):
    """League per-possession rates of the prior season (box_games)."""
    B = pd.read_parquet(CACHE / "box_games.parquet")
    b = B[B.season == season - 1]
    tot = b.poss.sum()
    r = {"pts": b.pts.sum() / tot, "reb": (b.oreb + b.dreb).sum() / tot, "ast": b.ast.sum() / tot,
         "tpm": b.fg3m.sum() / tot, "stl": b.stl.sum() / tot, "blk": b.blk.sum() / tot, "tov": b.tov.sum() / tot}
    return r


def fit_normal(mu, y):
    """sd = a * mu^b, discretised normal, MLE."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    def nll(th):
        sd = np.exp(th[0]) * np.maximum(mu, .3) ** th[1]
        return -np.log(np.clip(norm.cdf(y + .5, mu, sd) - norm.cdf(y - .5, mu, sd), 1e-9, 1)).sum()
    th = minimize(nll, [0.0, 0.5], method="Nelder-Mead").x
    return float(np.exp(th[0])), float(th[1])


def p_over(line, mu, dist):
    """P(Y > line) for half-point lines under dist = ("nb", k) or ("normal", a, b)."""
    from scipy.stats import norm
    if dist[0] == "normal":
        return 1 - norm.cdf(line, mu, dist[1] * np.maximum(mu, .3) ** dist[2])
    return 1 - nb_cdf(np.floor(line).astype(int), mu, dist[1])


def fit_k(mu, y):
    ks = np.exp(np.linspace(np.log(0.3), np.log(500), 120))
    ll = [nb_logpmf(y, mu, k).sum() for k in ks]
    return float(ks[int(np.argmax(ll))])


def backtest(season=2026):
    data = pi.BookerData()
    B = pd.read_parquet(CACHE / "box_games.parquet")
    B = B[B.season == season].copy()
    B["reb"] = B.oreb + B.dreb
    B = B.rename(columns={"fg3m": "tpm"})
    B = season_games(data, season, B)
    P = _priors(season)
    ppm = _ppm(data, season)
    lg_prev = _league_prev(season)
    dates = sorted(B.date.unique()); mid = dates[len(dates) // 2]
    # tune the in-season knobs on the FIRST half only (PRA MAE), then score the second half
    T1 = B[(B.poss >= 5) & (B.date < mid)]
    best = None
    for km in (0.15, 0.3, 0.6, 1.0):
        for gp_ in (2.0, 5.0):
            for era in (False, True):
                Mt = project_games(data, season, B, T1, P, ppm, km, gp_, era, lg_prev)
                Yt = B.set_index(["GAME_ID", "PLAYER_ID"])
                Mt = Mt.join(Yt[["pts", "reb", "ast"]], on=["GAME_ID", "PLAYER_ID"], rsuffix="_y")
                mae = float(np.mean(np.abs(Mt.pra - (Mt.pts_y + Mt.reb_y + Mt.ast_y))))
                print(f"  tune k_mult={km} g_prior={gp_} era={era}: first-half PRA MAE {mae:.3f}", flush=True)
                if best is None or mae < best[0]:
                    best = (mae, km, gp_, era)
    _, km, gp_, era = best
    print(f"  chosen: k_mult={km} g_prior={gp_} era={era}")
    M = project_games(data, season, B, B[B.poss >= 5], P, ppm, km, gp_, era, lg_prev)
    Y = B.set_index(["GAME_ID", "PLAYER_ID"])
    M = M.join(Y[STATS], on=["GAME_ID", "PLAYER_ID"], rsuffix="_y")
    M["pra_y"] = M.pts_y + M.reb_y + M.ast_y
    # baselines: last season per-game (bbref) and season-to-date per-game average
    b = fp._box()
    last = b[b.season == season - 1].set_index("nm")
    nmap = {int(p): pi.norm_name(n) for p, n in zip(data.PLAYERS[season].PLAYER_ID, data.PLAYERS[season].NAME)}
    col = {"pts": "total_points", "reb": "total_totalRb", "ast": "total_assists", "tpm": "total_threeFg",
           "stl": "total_steals", "blk": "total_blocks", "tov": "total_turnovers"}
    for s, c in col.items():
        M[s + "_last"] = [float(last.loc[nmap.get(p)][c] / last.loc[nmap.get(p)].games) if nmap.get(p) in last.index
                          and np.ndim(last.loc[nmap.get(p)][c]) == 0 else np.nan for p in M.PLAYER_ID]
    M["pra_last"] = M.pts_last + M.reb_last + M.ast_last
    Bs = B.sort_values("date")
    for s in STATS + ["pra"]:
        v = Bs.pts + Bs.reb + Bs.ast if s == "pra" else Bs[s]
        cum = v.groupby(Bs.PLAYER_ID).cumsum() - v
        cnt = Bs.groupby("PLAYER_ID").cumcount()
        Bs[s + "_std"] = np.where(cnt > 0, cum / cnt.replace(0, np.nan), np.nan)
    M = M.merge(Bs[["GAME_ID", "PLAYER_ID"] + [s + "_std" for s in STATS + ["pra"]]], on=["GAME_ID", "PLAYER_ID"])
    tr, te = M[M.date < mid], M[M.date >= mid].dropna(subset=[s + "_last" for s in STATS])
    res = {"season": season, "n_test": int(len(te)), "mid": mid, "stats": {},
           "params": {"k_mult": km, "g_prior": gp_, "era": bool(era)}}
    # early season (first 20 team games): where the prior matters most
    early = M[M.date < dates[min(len(dates) - 1, 40)]].dropna(subset=["pts_last"])
    res["early"] = {s: {"mae_model": round(float(np.mean(np.abs(early[s] - early[s + "_y"]))), 3),
                        "mae_last_season": round(float(np.nanmean(np.abs(early[s + "_last"] - early[s + "_y"]))), 3),
                        "mae_season_to_date": round(float(np.nanmean(np.abs(early[s + "_std"] - early[s + "_y"]))), 3)}
                    for s in STATS + ["pra"]}
    print("  early season MAE (model | last season | to-date):",
          {s: (v["mae_model"], v["mae_last_season"], v["mae_season_to_date"]) for s, v in res["early"].items()})
    for s in STATS + ["pra"]:
        # mean calibration fit on the FIRST half (projections ran ~2-4% low: possessions and
        # rates both), applied to the held-out second half and stored for live use
        cal_m = float(tr[s + "_y"].mean() / tr[s].mean())
        y = (tr[s + "_y"]).round().astype(int).values
        k = fit_k(tr[s].values * cal_m, y)
        # family choice INSIDE the first half: fit on its first half, Brier of over/under at the
        # near-projection line on its second half; the held-out second season half is untouched
        q = sorted(tr.date.unique()); qm = q[len(q) // 2]
        a_, b_ = tr[tr.date < qm], tr[tr.date >= qm]
        mu_b, y_b = b_[s].values * cal_m, b_[s + "_y"].values
        ln_b = np.floor(mu_b) + .5
        cand = {"nb": ("nb", fit_k(a_[s].values * cal_m, a_[s + "_y"].round().astype(int).values)),
                "normal": ("normal",) + fit_normal(a_[s].values * cal_m, a_[s + "_y"].values)}
        brier = {nm: float(np.mean((p_over(ln_b, mu_b, d) - (y_b > ln_b)) ** 2)) for nm, d in cand.items()}
        fam = min(brier, key=brier.get)
        dist = ("nb", k) if fam == "nb" else ("normal",) + fit_normal(tr[s].values * cal_m, tr[s + "_y"].values)
        yt = te[s + "_y"].round().astype(int).values
        mu = te[s].values * cal_m
        line = np.floor(mu) + 0.5                     # the line a book would hang near the projection
        po = p_over(line, mu, dist)
        over = (yt > line).astype(int)
        if dist[0] == "nb":
            lo = np.array([np.searchsorted(np.cumsum(np.exp(nb_logpmf(np.arange(int(m * 4 + 30)), m, k))), 0.1) for m in mu])
            hi = np.array([np.searchsorted(np.cumsum(np.exp(nb_logpmf(np.arange(int(m * 4 + 30)), m, k))), 0.9) for m in mu])
        else:
            sd_ = dist[1] * np.maximum(mu, .3) ** dist[2]
            lo, hi = np.round(mu - 1.2816 * sd_), np.round(mu + 1.2816 * sd_)
        bins = pd.cut(po, [0, .35, .45, .55, .65, 1.0])
        cal = pd.DataFrame({"b": bins, "p": po, "o": over}).groupby("b", observed=True).agg(p=("p", "mean"), o=("o", "mean"), n=("o", "size"))
        res["stats"][s] = {
            "k": round(k, 2), "cal": round(cal_m, 4), "dist": [dist[0]] + [round(float(x), 4) for x in dist[1:]],
            "brier_choice": {kk: round(v, 4) for kk, v in brier.items()},
            "brier_test": round(float(np.mean((po - over) ** 2)), 4),
            "mae_model": round(float(np.mean(np.abs(mu - yt))), 3),
            "mae_last_season": round(float(np.nanmean(np.abs(te[s + "_last"] - yt))), 3),
            "mae_season_to_date": round(float(np.nanmean(np.abs(te[s + "_std"] - yt))), 3),
            "bias": round(float(np.mean(mu - yt)), 3),
            "cov80": round(float(np.mean((yt >= lo) & (yt <= hi))), 3),
            "p_over_mean": round(float(po.mean()), 3), "over_rate": round(float(over.mean()), 3),
            "calibration": [[round(float(a), 3), round(float(b_), 3), int(c)] for a, b_, c in zip(cal.p, cal.o, cal.n)],
        }
        r = res["stats"][s]
        print(f"  {s:4s} {r['dist'][0]:6s} MAE model {r['mae_model']:.3f} | last-season {r['mae_last_season']:.3f} | "
              f"to-date {r['mae_season_to_date']:.3f}  bias {r['bias']:+.3f}  cov80 {r['cov80']:.2f}  "
              f"P(over) {r['p_over_mean']:.3f} vs {r['over_rate']:.3f}", flush=True)
    json.dump(res, open(CACHE / "props_validation.json", "w"), indent=1)
    print(f"backtest {season}: {res['n_test']} second-half player-games (k fit on games before {mid})")
    return res


def live(season=2027, days=7):
    from . import box_games as bg
    data = pi.BookerData()
    val = json.load(open(CACHE / "props_validation.json"))
    P = fp.project(fp._box(), season, {k: KMIN.get(k, 600) for k in fp.COUNT}, {"fg": 1000, "ft": 400},
                   fp.fit_age(fp._box(), range(2019, season), {k: KMIN.get(k, 600) for k in fp.COUNT}, {"fg": 1000, "ft": 400}))
    ppm = _ppm(data, season)
    # season to date: ESPN counts + on-court possessions from the stints
    to_date = pd.DataFrame(columns=["GAME_ID", "PLAYER_ID", "poss", "date", "team", "opp"] + STATS)
    bp = CACHE / f"espn_box_{season}.csv"
    if bp.exists() and season in data.STINTS:
        e = pd.read_csv(bp)
        e = e[e["min"] > 0].merge(bg.player_game_poss(data, season), on=["GAME_ID", "PLAYER_ID"], how="inner")
        to_date = season_games(data, season, e[["GAME_ID", "PLAYER_ID", "poss"] + STATS])
    g = data.GAMES[season]
    g = g[(g.SEASON_TYPE == "Regular Season") & g.HOME_WIN.isna()]
    if "PLACEHOLDER" in g:
        g = g[g.PLACEHOLDER != 1]
    first = g.DATE.min()
    g = g[g.DATE <= (pd.Timestamp(first) + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")]
    # who plays: projected rotation (minutes model) minus players listed OUT
    mp = pd.read_csv(CACHE / "minutes_proj.csv")
    mp = mp[(mp.season == season) & (mp.proj_pres >= 0.05)]
    status = {}
    try:
        from . import injury_availability as ia
        L = ia.load_log().sort_values("snapshot_utc").drop_duplicates("athlete_id", keep="last")
        L = L[L.event.astype(str) != "CLEARED"]
        for r in L.itertuples():
            status[pi.norm_name(r.name)] = str(r.designation or r.status).lower()
    except Exception as exc:
        print(f"  injury log unavailable ({exc})")
    rows = []
    for r in g.itertuples():
        for team, opp in ((r.HOME, r.AWAY), (r.AWAY, r.HOME)):
            for q in mp[mp.team == team].itertuples():
                stt = status.get(pi.norm_name(q.name), "")
                if stt == "out":
                    continue
                rows.append({"GAME_ID": int(r.GAME_ID), "PLAYER_ID": int(q.PLAYER_ID), "date": r.DATE, "team": team,
                             "opp": opp, "name": q.name, "status": stt})
    T = pd.DataFrame(rows)
    pz = val.get("params", {})
    M = project_games(data, season, to_date, T, P, ppm, pz.get("k_mult"), pz.get("g_prior"), pz.get("era"),
                      _league_prev(season)).merge(T[["GAME_ID", "PLAYER_ID", "name", "status"]], on=["GAME_ID", "PLAYER_ID"])
    # NOTE: no team-minutes scaling. Minutes are CONDITIONAL on playing (props void on DNPs);
    # summed over every rotation candidate they exceed 240 because ~10-15% sit on a given night,
    # but over those who play they are ~unbiased (backtest: possessions -1.6%).
    M = M[M["min"] >= 8].copy()
    for s in STATS:
        M[s] = M[s] * val["stats"][s]["cal"]
    M["pra"] = M.pts + M.reb + M.ast
    M.round(3).to_csv(CACHE / "props_live.csv", index=False)
    print(f"props live: {len(M)} player-games over {M.date.nunique()} dates ({M.date.min()}..{M.date.max()}); "
          f"k from backtest {val['season']}")
    return M


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        backtest(int(sys.argv[sys.argv.index("--backtest") + 1]) if len(sys.argv) > 2 else 2026)
    else:
        live()
