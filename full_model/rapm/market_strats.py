"""
Betting-strategy search vs the closing line, honest version.

Basis: completed games, model_p_home = ACTIVES probability (who suited up; projected shares,
never realized minutes) -- the inactive list is public before close, so this is bettable.
Market: vig-free closing consensus (SBR/oddsdata/Pinnacle; ESPN spreads -> prob for
2023-24 and 2024-25, synthetic 4% overround prices there).

"Star out": a player rated >= STAR_NET per 100 entering the game (Kalman state, pregame)
with a regular role (projected share >= STAR_SHARE) did not play. Uses no outcome info.

Strategies (all thresholds chosen WALK-FORWARD: tuned on seasons < s, applied to s):
  A  bet the model side every game
  B  A, excluding games with a star out (either team)
  C  bet only when |model - market| > tau
  D  C, excluding star-out games
  E  market+model logistic blend (fit on prior seasons), bet when |blend - market| > tau
Reports ROI, bets, bootstrap 90% CI (game-resampled), and the model-minus-market log-loss
by subgroup (does the model lose more to the market in star-out games?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache"
STAR_NET, STAR_SHARE = 2.5, 0.45
SEASONS = range(2018, 2027)
rng = np.random.default_rng(7)


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def payout(ml):          # profit per $1 staked if the bet wins
    return np.where(ml < 0, 100 / -ml, ml / 100)


def load():
    g = pd.read_csv(CACHE / "game_predictions_all.csv")
    rows = []
    for s in SEASONS:
        o = pd.read_csv(CACHE / f"odds_{s}.csv")
        o = o.rename(columns={"DATE": "date", "HOME": "home", "AWAY": "away"})
        if "ML_HOME" not in o:
            o["ML_HOME"] = o["ML_AWAY"] = np.nan
        if o.ML_HOME.isna().any():
            q = o.P_HOME.clip(.01, .99)
            am = lambda x: np.where(x >= .5, -100 * x / (1 - x), 100 * (1 - x) / x)
            miss = o.ML_HOME.isna()
            o.loc[miss, "ML_HOME"] = am(q[miss] * 1.04)
            o.loc[miss, "ML_AWAY"] = am((1 - q[miss]) * 1.04)
        sg = pd.read_csv(CACHE / f"seq_games_{s}.csv")[["GAME_ID", "date", "home", "away"]]
        pf = pd.read_parquet(CACHE / f"seq_players_{s}.parquet")
        pf["star"] = ((pf.o + pf.d) >= STAR_NET) & (pf.share >= STAR_SHARE)
        so = pf[pf.star].groupby(["GAME_ID", "team"]).agg(stars=("star", "size"), out=("played", lambda x: int((x == 0).sum()))).reset_index()
        so = so.merge(sg, on="GAME_ID")
        so["side"] = np.where(so.team == so.home, "h", "a")
        w = so.pivot_table(index="GAME_ID", columns="side", values="out", aggfunc="sum").fillna(0)
        w.columns = [f"stars_out_{c}" for c in w.columns]
        sg = sg.merge(w.reset_index(), on="GAME_ID", how="left").fillna({"stars_out_h": 0, "stars_out_a": 0})
        gs = g[g.season == s].merge(sg, on=["date", "home", "away"], how="left")
        gs = gs.merge(o[["date", "home", "away", "ML_HOME", "ML_AWAY"]], on=["date", "home", "away"], how="left")
        rows.append(gs)
    d = pd.concat(rows, ignore_index=True).dropna(subset=["market_p_home", "model_p_home", "ML_HOME"])
    d = d[d.calib_in_sample == 0]            # 2018 calibration is in-sample: never score it
    d["star_out"] = (d.stars_out_h + d.stars_out_a) > 0
    d["avail_shift"] = (d.margin_act - d.margin_strict).abs()
    d["edge"] = d.model_p_home - d.market_p_home
    return d


def bets(d, mask, p_col="model_p_home"):
    x = d[mask]
    home = x[p_col] > x.market_p_home
    won = np.where(home, x.home_win == 1, x.home_win == 0)
    pay = np.where(home, payout(x.ML_HOME.values), payout(x.ML_AWAY.values))
    return np.where(won, pay, -1.0), x.season.values


def dog_control(d, mask, p_col="model_p_home"):
    """Same games, always back the market underdog: isolates favourite-longshot exposure."""
    x = d[mask]
    home = x.market_p_home < 0.5
    won = np.where(home, x.home_win == 1, x.home_win == 0)
    pay = np.where(home, payout(x.ML_HOME.values), payout(x.ML_AWAY.values))
    return np.where(won, pay, -1.0)


def summary(name, prof):
    if len(prof) == 0:
        return {"strategy": name, "bets": 0}
    bs = [rng.choice(prof, len(prof)).mean() for _ in range(2000)]
    return {"strategy": name, "bets": len(prof), "roi": prof.mean(),
            "ci90_lo": np.quantile(bs, .05), "ci90_hi": np.quantile(bs, .95)}


def wf_tau(d, mask_fn, taus=(0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10), p_col="model_p_home", min_bets=150):
    """Walk-forward: tau for season s = best pooled ROI over seasons < s (>= min_bets)."""
    out, picks = [], {}
    for s in sorted(d.season.unique()):
        past = d[d.season < s]
        if past.season.nunique() < 2:
            continue
        best, bt = -9, 0.0
        for t in taus:
            pr, _ = bets(past, mask_fn(past) & ((past[p_col] - past.market_p_home).abs() > t), p_col)
            if len(pr) >= min_bets and pr.mean() > best:
                best, bt = pr.mean(), t
        cur = d[d.season == s]
        pr, _ = bets(cur, mask_fn(cur) & ((cur[p_col] - cur.market_p_home).abs() > bt), p_col)
        out.append(pr); picks[int(s)] = bt
    return np.concatenate(out) if out else np.array([]), picks


def blend_probs(d):
    """Walk-forward logistic: y ~ logit(market) + logit(model) [+ star_out interaction]."""
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    X = np.c_[lg(d.market_p_home), lg(d.model_p_home) - lg(d.market_p_home),
              (lg(d.model_p_home) - lg(d.market_p_home)) * d.star_out]
    pb = np.full(len(d), np.nan); coefs = {}
    for s in sorted(d.season.unique()):
        tr, te = (d.season < s).values, (d.season == s).values
        if d[tr].season.nunique() < 2:
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X[tr], d.home_win[tr])
        pb[te] = m.predict_proba(X[te])[:, 1]; coefs[int(s)] = np.round(m.coef_[0], 3)
    return pb, coefs


def main():
    d = load()
    print(f"games with model + closing market: {len(d)}  ({d.season.min()}-{d.season.max()})")
    print(f"star-out games: {d.star_out.mean():.1%}  (stars defined: net >= {STAR_NET}, share >= {STAR_SHARE})\n")

    print("model minus market log-loss (positive = market better), by subgroup.")
    print("PRE-GAME basis (the actives basis already knows the star sat -> tautological):")
    d["dll"] = ll(d.model_p_home_pregame, d.home_win) - ll(d.market_p_home, d.home_win)
    for nm, m in [("all", d.index == d.index), ("no star out", ~d.star_out), ("star out", d.star_out),
                  ("avail shift > 2 pts", d.avail_shift > 2), ("avail shift <= 2", d.avail_shift <= 2)]:
        x = d[m]
        se = x.dll.std() / np.sqrt(len(x))
        print(f"  {nm:22s} n={len(x):5d}  dLL {x.dll.mean():+.4f} (se {se:.4f})")
    print("  by season (dLL, no-star-out / star-out):")
    for s, x in d.groupby("season"):
        print(f"    {s}: {x[~x.star_out].dll.mean():+.4f} / {x[x.star_out].dll.mean():+.4f}")

    res = []
    pr, _ = bets(d, d.index == d.index); res.append(summary("A every game, model side", pr))
    pr, _ = bets(d, ~d.star_out); res.append(summary("B excl. star-out games", pr))
    for t in (0.03, 0.05, 0.08):
        pr, _ = bets(d, d.edge.abs() > t); res.append(summary(f"C |edge|>{t:.2f} (fixed, in-sample)", pr))
        pr, _ = bets(d, (d.edge.abs() > t) & ~d.star_out); res.append(summary(f"D |edge|>{t:.2f} excl star-out (fixed)", pr))
    pr, pk = wf_tau(d, lambda x: x.index == x.index); res.append(summary("C walk-forward tau", pr)); print("\nC tau picks:", pk)
    pr, pk = wf_tau(d, lambda x: ~x.star_out); res.append(summary("D walk-forward tau, excl star-out", pr)); print("D tau picks:", pk)
    d["p_blend"], coefs = blend_probs(d)
    print("blend coefs [logit mkt, model-mkt gap, gap x star_out]:", coefs)
    e = d.dropna(subset=["p_blend"])
    print(f"blend LL {ll(e.p_blend, e.home_win).mean():.4f} vs market {ll(e.market_p_home, e.home_win).mean():.4f} on {len(e)} OOS games")
    pr, pk = wf_tau(e, lambda x: x.index == x.index, p_col="p_blend", taus=(0.0, 0.01, 0.02, 0.03, 0.05))
    res.append(summary("E blend, walk-forward tau", pr)); print("E tau picks:", pk)
    for t in (0.05, 0.08):
        for basis in ("model_p_home", "model_p_home_pregame"):
            mk = (d[basis] - d.market_p_home).abs() > t
            pr, _ = bets(d, mk, basis); ctl = dog_control(d, mk)
            res.append({**summary(f"{basis.replace('model_p_home', 'm') or 'm'} |edge|>{t:.2f} MINUS dog control", pr - ctl),
                        "note": f"model {pr.mean():+.3f} vs dog {ctl.mean():+.3f}"})
    r = pd.DataFrame(res)
    print("\n" + r.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nbreak-even is ~0 ROI after vig; a real edge needs ci90_lo > 0 under walk-forward selection.")
    by = d.assign(prof=bets(d, d.index == d.index)[0]).groupby("season").prof.mean()
    print("\nA by season:", by.round(3).to_dict())
    return d, r


if __name__ == "__main__":
    main()
