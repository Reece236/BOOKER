"""Where does BOOKER lose to the closing line, and does it add anything to it?

Benchmark: vig-removed CLOSING moneylines (cache/odds_{s}.csv). Model probabilities
are walk-forward (game_odds.py); DARKO margins come from compare_darko.game_frame.

1. STACKING  walk-forward logit(p) = a + b*logit(market) + sum c_k*logit(model_k):
             if the stack beats the market out of sample, the model carries
             information the close doesn't.
2. AVAILABILITY  gap (model - market log-loss) in games where a team's top-2
             projected-share player sat vs. games where every top-2 player played.
3. CALENDAR  gap by month.
4. DISAGREEMENT  when |model - market| is large, who was right.
Run: python market_gap.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

import compare_darko as cd

CACHE = cd.CACHE
EPS = 1e-6


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_stack(X, y):
    f = lambda b: -np.mean(y * np.log(expit(X @ b) + EPS) + (1 - y) * np.log(1 - expit(X @ b) + EPS))
    return minimize(f, np.r_[0.0, 1.0, np.zeros(X.shape[1] - 2)], method="BFGS").x


def stack(F, cols):
    out = pd.Series(np.nan, index=F.index)
    for s in sorted(F.season.unique()):
        tr, te = F[F.season < s], F[F.season == s]
        if len(tr) < 1500:
            continue
        mk = lambda d: np.column_stack([np.ones(len(d))] + [logit(np.clip(d[c], EPS, 1 - EPS)) for c in cols])
        b = fit_stack(mk(tr), tr.home_win.to_numpy())
        out[te.index] = expit(mk(te) @ b)
    return out


def missing_star(seasons):
    """{GAME_ID: n of the two teams whose top-2 projected-share player did NOT play}."""
    out = {}
    for s in seasons:
        P = pd.read_parquet(CACHE / f"seq_players_{s}.parquet")
        P["rk"] = P.groupby(["GAME_ID", "team"]).share.rank(ascending=False, method="first")
        top = P[P.rk <= 2]
        miss = top.groupby(["GAME_ID", "team"]).played.min().eq(0).groupby("GAME_ID").sum()
        out.update(miss.to_dict())
    return out


def main():
    D = cd.load()
    F = cd.game_frame(D)
    for c in ("m_bk_strict", "m_bk_actives", "m_dpm_actives"):
        F["p_" + c] = cd.walk_forward(F, c)
    F = F.dropna(subset=["market_p_home", "p_m_bk_strict", "p_m_bk_actives", "p_m_dpm_actives"]).reset_index(drop=True)
    y = F.home_win.to_numpy()
    print(f"games with closing lines + walk-forward model probs: {len(F)} "
          f"(seasons {sorted(F.season.unique())})")

    # 1. stacking
    F["stack_bk"] = stack(F, ["market_p_home", "p_m_bk_actives"])
    F["stack_bk_dk"] = stack(F, ["market_p_home", "p_m_bk_actives", "p_m_dpm_actives"])
    E = F.dropna(subset=["stack_bk", "stack_bk_dk"])
    ye = E.home_win.to_numpy()
    print(f"\n1. STACKING (scored on {len(E)} games, each season fit on prior seasons only)")
    for lab, c in (("market (close)", "market_p_home"), ("BOOKER actives", "p_m_bk_actives"),
                   ("market + BOOKER", "stack_bk"), ("market + BOOKER + DARKO", "stack_bk_dk")):
        print(f"  {lab:26s} logloss {ll(E[c], ye):.4f}")
    rng = np.random.default_rng(0); d = []
    for _ in range(1000):
        i = rng.integers(0, len(E), len(E))
        d.append(ll(E.stack_bk.values[i], ye[i]) - ll(E.market_p_home.values[i], ye[i]))
    print("  (market+BOOKER) - market 95%% CI [%+.4f, %+.4f]" % tuple(np.percentile(d, [2.5, 97.5])))
    Xall = np.column_stack([np.ones(len(F)), logit(F.market_p_home.clip(EPS, 1 - EPS)), logit(F.p_m_bk_actives.clip(EPS, 1 - EPS))])
    b = fit_stack(Xall, y)
    print(f"  pooled stack weights: market {b[1]:.2f}, BOOKER {b[2]:.2f}")

    # 2. availability
    ms = missing_star(sorted(F.season.unique()))
    F["miss"] = F.GAME_ID.map(ms).fillna(0)
    print("\n2. AVAILABILITY (top-2 projected-share player out for either team)")
    for lab, m in (("all top-2 played", F.miss == 0), ("a top-2 player sat", F.miss > 0)):
        g = F[m]; yy = g.home_win.to_numpy()
        print(f"  {lab:20s} n={len(g):5d} ({len(g)/len(F):.0%}) | market {ll(g.market_p_home, yy):.4f} | "
              f"strict {ll(g.p_m_bk_strict, yy):.4f} ({ll(g.p_m_bk_strict, yy)-ll(g.market_p_home, yy):+.4f}) | "
              f"actives {ll(g.p_m_bk_actives, yy):.4f} ({ll(g.p_m_bk_actives, yy)-ll(g.market_p_home, yy):+.4f})")

    # 3. calendar
    F["month"] = pd.to_datetime(F.date).dt.month
    mo = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    print("\n3. CALENDAR (actives - market log-loss)")
    print("  " + "  ".join(f"{mo.get(m, m)} {ll(g.p_m_bk_actives, g.home_win)-ll(g.market_p_home, g.home_win):+.4f}"
                           f"(n={len(g)})" for m, g in F.groupby("month") if m in mo))

    # 4. disagreement
    F["dis"] = (F.p_m_bk_actives - F.market_p_home).abs()
    print("\n4. DISAGREEMENT (|BOOKER actives - market|)")
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, 1)):
        g = F[(F.dis >= lo) & (F.dis < hi)]
        if len(g) < 30:
            continue
        side = np.where(g.p_m_bk_actives > g.market_p_home, g.home_win, 1 - g.home_win)
        mk = np.where(g.p_m_bk_actives > g.market_p_home, g.market_p_home, 1 - g.market_p_home)
        print(f"  {lo:.2f}-{hi:.2f}: n={len(g):5d} | BOOKER's side won {side.mean():.3f} vs market-implied {mk.mean():.3f} "
              f"| LL gap {ll(g.p_m_bk_actives, g.home_win)-ll(g.market_p_home, g.home_win):+.4f}")


if __name__ == "__main__":
    main()
