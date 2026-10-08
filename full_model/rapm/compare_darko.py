"""BOOKER vs DARKO on game-by-game, point-in-time data (no hindsight for either).

DARKO: cache/darko_games.parquet (data_ingest/fetch_darko_games.py) -- each game row
is that player's PRE-game DPM (timing verified there). BOOKER: forecast.sequential
(pre-game O/D after Kalman updates through the previous date) and the prior-only
BookerFormer row s-1. Both are weighted with the SAME minutes shares unless noted.

A. GAMES     walk-forward logistic P(home win) on [team-strength margin, form, rest,
             b2b]; strength = sum(share * rating) with
               strict  : our pre-game expected shares (nothing about tonight)
               actives : shares renormalized over who suits up (injury-report proxy)
               native  : DARKO weighted by its own pre-game x_minutes
B. PRESEASON each metric at the season opener x projected shares (minutes_share) ->
             team net -> wins via a walk-forward linear map (fit on prior seasons).
C. PLAYERS   rating as of a cutoff (opener, midseason) vs REST-OF-SEASON outcomes:
             prior-free pure RAPM on stints after the cutoff and real on-court +/-,
             players with >= 500 rest-of-season minutes. Spearman + bootstrap CI.

Writes cache/darko_comparison.csv. Run: python compare_darko.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "data_ingest"))
from forecast import player_impacts as pi                      # noqa: E402
from forecast import minutes_share as ms                       # noqa: E402
from forecast import game_odds as go                           # noqa: E402
from fetch_darko_games import darko_pregame                    # noqa: E402

CACHE = HERE / "cache"
SEASONS = list(range(2019, 2027))
EPS = 1e-12
DARKO_FILL = -2.0          # DARKO value for a rostered player DARKO doesn't carry


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def boot_diff(a, b, y, f, n=1000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        out.append(f(a[i], y[i]) - f(b[i], y[i]))
    return np.percentile(out, [2.5, 97.5])


# ------------------------------------------------------------------ data
def load():
    D = darko_pregame()
    tm = {}
    for s in SEASONS + [2027]:
        p = CACHE / f"teams_{s}.csv"
        if p.exists():
            t = pd.read_csv(p); tm.update(dict(zip(t.TEAM_ID.astype(int), t.ABBR)))
    D["team"] = D.tm_id.map(lambda x: tm.get(int(x)) if x == x else None)
    return D


# ------------------------------------------------------------------ A. games
def game_frame(D):
    rows = []
    for s in SEASONS:
        G = pd.read_csv(CACHE / f"seq_games_{s}.csv")
        P = pd.read_parquet(CACHE / f"seq_players_{s}.parquet")
        d = D[D.season == s]
        dk = dict(zip(zip(d.date, d.nba_id), d.dpm_pre))
        P["dpm"] = [dk.get((dt, p), DARKO_FILL) for dt, p in zip(P.date, P.PLAYER_ID)]
        P["bk"] = P.o + P.d
        def strength(g, col, mode):
            if mode == "actives":
                g = g[g.played == 1]
                w = g.share * (5.0 / g.share.sum()) if g.share.sum() > 0 else g.share
                return float((w * g[col]).sum())
            fill = max(0.0, 5.0 - g.share.sum())
            return float((g.share * g[col]).sum() + fill * (ms.load(s).repl.iloc[0] if col == "bk" else DARKO_FILL / 2))
        agg = {}
        for (gid, team), g in P.groupby(["GAME_ID", "team"]):
            agg[(gid, team)] = {f"{c}_{m}": strength(g, c, m) for c in ("bk", "dpm") for m in ("strict", "actives")}
        # DARKO-native: its own pre-game x_minutes over its roster rows that date
        nat = d.assign(w=d.x_minutes_pre.fillna(0)).groupby(["date", "team"]).apply(
            lambda x: 5.0 * (x.dpm_pre * x.w).sum() / max(x.w.sum(), 1e-9), include_groups=False)
        for r in G.itertuples():
            h, a = agg.get((r.GAME_ID, r.home)), agg.get((r.GAME_ID, r.away))
            if not h or not a:
                continue
            row = {"season": s, "GAME_ID": r.GAME_ID, "date": r.date, "home": r.home, "away": r.away}
            for k in h:
                row["m_" + k] = h[k] - a[k]
            row["m_dpm_native"] = nat.get((r.date, r.home), np.nan) - nat.get((r.date, r.away), np.nan)
            rows.append(row)
    F = pd.DataFrame(rows)
    # shared context features + outcome from the production game_odds frame
    ctx = pd.read_csv(CACHE / "game_predictions_all.csv")[
        ["season", "date", "home", "away", "home_win", "formEW", "rest_diff", "b2b_home",
         "b2b_away", "market_p_home"]]
    return F.merge(ctx, on=["season", "date", "home", "away"])


def walk_forward(F, mcol):
    feats = [mcol, "formEW", "rest_diff", "b2b_home", "b2b_away"]
    p = pd.Series(np.nan, index=F.index)
    for s in sorted(F.season.unique()):
        tr = F[(F.season < s)].dropna(subset=feats)
        if not len(tr):
            continue
        m = F.season == s
        ok = m & F[feats].notna().all(1)
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(tr[feats], tr.home_win)
        p[ok] = lr.predict_proba(F.loc[ok, feats])[:, 1]
    return p


def test_games(D):
    F = game_frame(D)
    cols = {"BOOKER strict": "m_bk_strict", "DARKO strict (our shares)": "m_dpm_strict",
            "BOOKER actives": "m_bk_actives", "DARKO actives (our shares)": "m_dpm_actives",
            "DARKO native (x_minutes)": "m_dpm_native"}
    for k, c in cols.items():
        F["p_" + c] = walk_forward(F, c)
    for mode in ("strict", "actives"):
        F[f"m_blend_{mode}"] = 0.5 * F[f"m_bk_{mode}"] / F[f"m_bk_{mode}"].std() \
            + 0.5 * F[f"m_dpm_{mode}"] / F[f"m_dpm_{mode}"].std()
        F[f"p_m_blend_{mode}"] = walk_forward(F, f"m_blend_{mode}")
        cols[f"50/50 blend {mode}"] = f"m_blend_{mode}"
    E = F[F.season >= 2020].dropna(subset=["p_" + c for c in cols.values()])
    out = []
    print(f"\nA. GAMES (walk-forward, scored 2020-2026, n={len(E)})")
    for k, c in cols.items():
        out.append(("games", k, "logloss_all", ll(E["p_" + c], E.home_win), len(E)))
    M = E.dropna(subset=["market_p_home"])
    for k, c in cols.items():
        out.append(("games", k, "logloss_market_games", ll(M["p_" + c], M.home_win), len(M)))
    out.append(("games", "MARKET", "logloss_market_games", ll(M.market_p_home, M.home_win), len(M)))
    for r in out:
        print(f"  {r[1]:28s} {r[2]:22s} {r[3]:.4f}  (n={r[4]})")
    for mode in ("strict", "actives"):
        lo, hi = boot_diff(E[f"p_m_bk_{mode}"].values, E[f"p_m_dpm_{mode}"].values,
                           E.home_win.values, ll)
        print(f"  BOOKER - DARKO {mode} logloss diff 95% CI [{lo:+.4f}, {hi:+.4f}]  (negative = BOOKER better)")
    return out


# ------------------------------------------------------------------ B. preseason
def test_preseason(D):
    rat = pd.read_csv(HERE / "booker_bookerformer_ratings.csv")
    data = pi.BookerData()
    rows = []
    for s in SEASONS:
        proj = ms.load(s)
        first = D[D.season == s].sort_values("date").groupby("nba_id").first()   # opener, pre-game
        dk = first.dpm_pre.to_dict()
        r1 = rat[rat.season == s - 1]
        bk = dict(zip(r1.PLAYER_ID, r1.impact_total))
        pre = pd.read_csv(CACHE / f"preseason_{s}.csv").set_index("team")
        n_bk = ms.team_nets(bk, s, proj=proj)
        n_dk = ms.team_nets({int(k): v for k, v in dk.items()}, s, proj=proj, repl=DARKO_FILL / 2)
        for t in n_bk:
            aw = data.ACTUAL_WINS.get((t, s))
            gp = 72 if s in (2020, 2021) else 82
            if aw is None:
                continue
            rows.append({"season": s, "team": t, "bk": n_bk[t], "dk": n_dk.get(t, np.nan),
                         "booker_sim": pre.sim_mean.get(t, np.nan), "pct": aw / gp, "gp": gp})
    T = pd.DataFrame(rows)
    T["blend"] = 0.5 * T.bk / T.bk.std() + 0.5 * T.dk / T.dk.std()
    res = []
    print("\nB. PRESEASON WINS (projected rosters + shares for both; walk-forward net->win% map)")
    for col in ("bk", "dk", "blend"):
        err = []
        for s in SEASONS[1:]:
            tr, te = T[T.season < s], T[T.season == s]
            b, a = np.polyfit(tr[col], tr.pct, 1)
            err += list((a + b * te[col] - te.pct) * 82)
        name = {"bk": "BOOKER (row s-1)", "dk": "DARKO (opener)", "blend": "50/50 blend"}[col]
        rm = float(np.sqrt(np.mean(np.square(err))))
        res.append(("preseason", name, "wins_rmse_per82_2020_26", rm, len(err)))
        print(f"  {name:22s} wins RMSE/82 {rm:.2f}")
    te = T[T.season >= 2020]
    e = (te.booker_sim / te.gp - te.pct) * 82
    print(f"  {'BOOKER full sim':22s} wins RMSE/82 {np.sqrt(np.mean(e**2)):.2f}  (production preseason.py)")
    res.append(("preseason", "BOOKER full sim", "wins_rmse_per82_2020_26", float(np.sqrt(np.mean(e ** 2))), len(e)))
    return res


# ------------------------------------------------------------------ C. players
def rest_of_season(data, s, cutoff):
    st = data.STINTS[s]
    g = data.GAMES[s]
    dt = dict(zip(g.GAME_ID.astype("int64"), g.DATE))
    st = st[st.GAME_ID.astype(str).str.startswith("2")]
    st = st[st.GAME_ID.astype("int64").map(dt) >= cutoff]
    ids = sorted({p for l in st.home for p in l} | {p for l in st.away for p in l})
    col = {p: i for i, p in enumerate(ids)}
    r, c, v = [], [], []
    for i, (h, a) in enumerate(zip(st.home, st.away)):
        for p in h: r.append(i); c.append(col[p]); v.append(1.0)
        for p in a: r.append(i); c.append(col[p]); v.append(-1.0)
    X = csr_matrix((v, (r, c)), shape=(len(st), len(ids)))
    pure = Ridge(alpha=2800.0).fit(X, st.Y, sample_weight=st.POSS).coef_
    sec, pm, poss = {}, {}, {}
    for h, a, du, x, ps in zip(st.home, st.away, st.DURATION_SECONDS, st.PLUS_MINUS, st.POSS):
        for p, sg in [(p, 1) for p in h] + [(p, -1) for p in a]:
            sec[p] = sec.get(p, 0) + du; pm[p] = pm.get(p, 0) + sg * x; poss[p] = poss.get(p, 0) + ps
    return pd.DataFrame({"PLAYER_ID": ids, "pure": pure,
                         "min": [sec[p] / 60 for p in ids],
                         "onoff": [100 * pm[p] / poss[p] for p in ids]})


def test_players(D):
    data = pi.BookerData()
    rat = pd.read_csv(HERE / "booker_bookerformer_ratings.csv")
    rows = []
    for s in SEASONS:
        g = data.GAMES[s]; g = g[g.SEASON_TYPE == "Regular Season"]
        dates = sorted(g.DATE.unique())
        P = pd.read_parquet(CACHE / f"seq_players_{s}.parquet")
        P["bk_seq"] = P.o + P.d
        for lab, cut in (("opener", dates[0]), ("midseason", dates[len(dates) // 2])):
            seq = P[P.date >= cut].sort_values("date").groupby("PLAYER_ID").bk_seq.first()   # pre-game state at cutoff
            dk = D[(D.season == s) & (D.date >= cut)].sort_values("date").groupby("nba_id").dpm_pre.first()
            r1 = rat[rat.season == s - 1].set_index("PLAYER_ID").impact_total
            out = rest_of_season(data, s, cut)
            out = out[out["min"] >= 500].copy()
            out["BOOKER_seq"] = out.PLAYER_ID.map(seq)
            out["DARKO"] = out.PLAYER_ID.map(dk)
            out["BookerFormer_prior"] = out.PLAYER_ID.map(r1)
            out["season"], out["cut"] = s, lab
            rows.append(out)
    # identical player sample for every model: rated by all three (rookies excluded --
    # BookerFormer has no prior row for them, so including them only for some models
    # would score the models on different populations)
    R = pd.concat(rows).dropna(subset=["BOOKER_seq", "DARKO", "BookerFormer_prior"])
    res = []
    print("\nC. PLAYERS (rating at cutoff vs rest-of-season; >=500 later minutes; Spearman)")
    for lab, g in R.groupby("cut"):
        for arb in ("pure", "onoff"):
            vals = {m: spearmanr(g[m], g[arb], nan_policy="omit")[0]
                    for m in ("BOOKER_seq", "DARKO", "BookerFormer_prior")}
            sp = lambda a, y: spearmanr(a, y)[0]
            lo, hi = boot_diff(g.BOOKER_seq.values, g.DARKO.values, g[arb].values, sp, n=500)
            print(f"  {lab:9s} vs {arb:5s} (n={len(g)}): " + "  ".join(f"{k} {v:.3f}" for k, v in vals.items())
                  + f"  | BOOKER-DARKO CI [{lo:+.3f}, {hi:+.3f}]")
            for k, v in vals.items():
                res.append(("players", k, f"{lab}_vs_{arb}_spearman", v, len(g)))
    return res


if __name__ == "__main__":
    D = load()
    out = test_games(D) + test_preseason(D) + test_players(D)
    pd.DataFrame(out, columns=["test", "model", "metric", "value", "n"]).to_csv(
        CACHE / "darko_comparison.csv", index=False)
    print("\nwrote cache/darko_comparison.csv")
