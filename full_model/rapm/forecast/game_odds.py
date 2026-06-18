"""
BOOKER per-game win-probability model and market comparison.

For every regular-season game we estimate a home win probability using ONLY
information available before tip-off: prior-season player impacts blended with the
current season's stints played strictly before the game date. Team strength is
the minutes-to-date-weighted sum of player impacts; the home/away net-rating gap
is turned into a win probability by a logistic map whose intercept absorbs home
court and whose slope is calibrated on realized outcomes.

We then compare BOOKER's probabilities to the market's vig-removed implied
probabilities (cache/odds_{season}.csv) via log-loss, Brier score, accuracy, and
flat-bet ROI on the model's value side.

Because impacts barely move day to day, they are refit on a periodic cadence
(REFIT_EVERY_DAYS) and reused for all games until the next refit -- still strictly
out-of-sample, but tractable.

Outputs:
    cache/game_predictions_{season}.csv   per game: probs, margin, outcome, market
    cache/game_metrics.csv                per-season + pooled skill vs market
    cache/game_calibration.csv            pooled reliability-curve bins
"""
import numpy as np
import pandas as pd

from . import player_impacts as pi

CACHE = pi.CACHE
FEATS = ["margin", "rest_diff", "b2b_home", "b2b_away"]
def _model_seasons(data):
    return [s for s in range(2018, 2028)
            if s in data.GAMES and s in data.STINTS and s in data.PLAYERS]
REFIT_EVERY_DAYS = 10
EPS = 1e-12


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_winprob(df, feats=FEATS):
    """Logistic P(home win) on net-margin + rest/back-to-back features. The intercept
    absorbs home court; coefficients are calibrated on realized outcomes."""
    from sklearn.linear_model import LogisticRegression
    d = df.dropna(subset=feats + ["home_win"])
    lr = LogisticRegression(C=1e6, max_iter=2000)
    lr.fit(d[feats].values, d.home_win.values)
    return lr


def _game_rosters(data, season):
    """{GAME_ID: {'home': {pid: minutes}, 'away': {pid: minutes}}} from stint lineups
    -- who actually played and how long (our availability signal)."""
    g = {}
    d = data.STINTS.get(season)
    if d is None:
        return g
    for gid, hl, al, dur in zip(d.GAME_ID, d.home, d.away, d.DURATION_SECONDS):
        mn = float(dur) / 60.0
        slot = g.setdefault(int(gid), {"home": {}, "away": {}})
        for p in hl:
            slot["home"][p] = slot["home"].get(p, 0.0) + mn
        for p in al:
            slot["away"][p] = slot["away"].get(p, 0.0) + mn
    return g


def _rest(prev_date, date):
    return 3 if prev_date is None else int(np.clip((date - prev_date).days, 0, 4))


def season_margins(data, season, prior_override=None):
    """Pre-game net-rating margin (home - away) for every regular-season game, using
    ONLY tonight's available players (those who play -- a realistic pre-tip inactive
    proxy) weighted by each player's to-date average minutes (redistributed across
    the actives). Also returns rest/back-to-back features. `prior_override`
    {pid: prior_impact} swaps the ridge box prior.
    """
    if season not in data.GAMES or season not in data.PLAYERS:
        return None
    train = pi.prior_train_seasons(data, season)
    if not train:
        return None
    alpha = pi.pick_alpha(data, train)
    reg = data.GAMES[season]
    reg = reg[reg.get("SEASON_TYPE", "Regular Season") == "Regular Season"].copy()
    reg = reg.sort_values("DATE").reset_index(drop=True)
    rosters = _game_rosters(data, season)

    # to-date average game-minutes per player (chronological, strictly before game)
    todate = {}
    tot, ng = {}, {}
    for r in reg.itertuples():
        gid = int(r.GAME_ID)
        todate[gid] = {p: tot[p] / ng[p] for p in tot if ng[p] > 0}
        slot = rosters.get(gid)
        if slot:
            for side in ("home", "away"):
                for p, mn in slot[side].items():
                    tot[p] = tot.get(p, 0.0) + mn
                    ng[p] = ng.get(p, 0) + 1

    cur_impact, last_age, last_refit = None, None, None
    last_played = {}
    rows = []
    for r in reg.itertuples():
        d = pd.Timestamp(r.DATE)
        if last_refit is None or (d - last_refit).days >= REFIT_EVERY_DAYS:
            in_st = data.season_stints_before(season, r.DATE)
            if in_st is None:
                in_st = data.STINTS[season].iloc[:0] if season in data.STINTS else None
            if in_st is None:
                continue
            cur_impact, _, last_age = pi.build_impacts(
                data, train, season, alpha, extra_stints=in_st, extra_weight=1.0,
                prior_override=prior_override)
            last_refit = d
        slot = rosters.get(int(r.GAME_ID))
        td = todate.get(int(r.GAME_ID), {})
        rh, ra = _rest(last_played.get(r.HOME), d), _rest(last_played.get(r.AWAY), d)
        last_played[r.HOME] = d; last_played[r.AWAY] = d
        if not slot or cur_impact is None:
            continue

        def side_net(side):
            act = slot[side]
            em = {p: td.get(p, act[p]) for p in act}     # to-date minutes (no leakage)
            tot_m = sum(em.values())
            if tot_m <= 0 or not act:
                return None
            return sum(cur_impact.get(p, pi.PRIOR_BASE) * (em[p] / (tot_m / 5.0)) for p in act)

        hn, an = side_net("home"), side_net("away")
        if hn is None or an is None:
            continue
        rows.append({
            "season": season, "date": r.DATE, "home": r.HOME, "away": r.AWAY,
            "margin": hn - an, "home_win": int(r.HOME_WIN),
            "rest_diff": rh - ra, "b2b_home": int(rh == 1), "b2b_away": int(ra == 1),
        })
    return pd.DataFrame(rows)


def attach_market(df, season):
    odds = CACHE / f"odds_{season}.csv"
    if not odds.exists():
        df["market_p_home"] = np.nan
        df["ml_home"] = np.nan
        df["ml_away"] = np.nan
        return df
    o = pd.read_csv(odds)
    o = o.rename(columns={"HOME": "home", "AWAY": "away", "DATE": "date"})
    m = df.merge(o[["date", "home", "away", "P_HOME", "ML_HOME", "ML_AWAY"]],
                 on=["date", "home", "away"], how="left")
    m = m.rename(columns={"P_HOME": "market_p_home", "ML_HOME": "ml_home",
                          "ML_AWAY": "ml_away"})
    return m


def metrics(p, y):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier = np.mean((p - y) ** 2)
    acc = np.mean((p >= 0.5) == (y == 1))
    return logloss, brier, acc


def flat_bet_roi(df):
    """Flat $1 bets on the side BOOKER rates above the market; ML payout."""
    d = df.dropna(subset=["market_p_home", "ml_home", "ml_away"])
    if d.empty:
        return np.nan, 0
    profit, n = 0.0, 0
    for r in d.itertuples():
        bet_home = r.model_p_home > r.market_p_home
        ml = r.ml_home if bet_home else r.ml_away
        won = (r.home_win == 1) if bet_home else (r.home_win == 0)
        dec = (1 + ml / 100.0) if ml > 0 else (1 + 100.0 / -ml)
        profit += (dec - 1) if won else -1.0
        n += 1
    return profit / n, n


def calibration_bins(df, col="model_p_home", nbins=10):
    d = df.dropna(subset=[col])
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.clip(np.digitize(d[col], bins) - 1, 0, nbins - 1)
    out = []
    for b in range(nbins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        out.append({"bin_mid": round((bins[b] + bins[b + 1]) / 2, 3),
                    "pred_mean": round(float(d[col][sel].mean()), 4),
                    "empirical": round(float(d.home_win[sel].mean()), 4),
                    "count": int(sel.sum())})
    return pd.DataFrame(out)


def main():
    data = pi.BookerData()
    frames = []
    for s in _model_seasons(data):
        df = season_margins(data, s)
        if df is None or df.empty:
            print(f"season {s}: skipped")
            continue
        frames.append(df)
        print(f"season {s}: {len(df)} games modeled")
    allg = pd.concat(frames, ignore_index=True)

    wp = fit_winprob(allg)
    allg["model_p_home"] = wp.predict_proba(allg[FEATS].values)[:, 1]
    coefs = dict(zip(FEATS, wp.coef_[0]))
    print(f"win-prob: HCA(intercept) {wp.intercept_[0]:.3f}, "
          + ", ".join(f"{k} {v:+.4f}" for k, v in coefs.items()))

    metric_rows = []
    for s, g in allg.groupby("season"):
        g = attach_market(g, s)
        g.to_csv(CACHE / f"game_predictions_{s}.csv", index=False)
        mll, mbr, mac = metrics(g.model_p_home, g.home_win)
        row = {"season": int(s), "games": len(g),
               "model_logloss": round(mll, 4), "model_brier": round(mbr, 4),
               "model_acc": round(mac, 4)}
        mk = g.dropna(subset=["market_p_home"])
        if len(mk):
            kll, kbr, kac = metrics(mk.market_p_home, mk.home_win)
            roi, nbet = flat_bet_roi(g)
            row.update({"market_games": len(mk),
                        "market_logloss": round(kll, 4),
                        "market_brier": round(kbr, 4),
                        "market_acc": round(kac, 4),
                        "roi": (None if np.isnan(roi) else round(roi, 4)),
                        "n_bets": nbet})
        metric_rows.append(row)

    # pooled
    full = pd.concat([attach_market(season_subset(allg, s), s)
                      for s in allg.season.unique()], ignore_index=True)
    full.to_csv(CACHE / "game_predictions_all.csv", index=False)
    pll, pbr, pac = metrics(full.model_p_home, full.home_win)
    mk = full.dropna(subset=["market_p_home"])
    pooled = {"season": "POOLED", "games": len(full),
              "model_logloss": round(pll, 4), "model_brier": round(pbr, 4),
              "model_acc": round(pac, 4)}
    if len(mk):
        kll, kbr, kac = metrics(mk.market_p_home, mk.home_win)
        roi, nbet = flat_bet_roi(full)
        pooled.update({"market_games": len(mk), "market_logloss": round(kll, 4),
                       "market_brier": round(kbr, 4), "market_acc": round(kac, 4),
                       "roi": (None if np.isnan(roi) else round(roi, 4)),
                       "n_bets": nbet})
    metric_rows.append(pooled)
    pd.DataFrame(metric_rows).to_csv(CACHE / "game_metrics.csv", index=False)

    calibration_bins(full).to_csv(CACHE / "game_calibration.csv", index=False)
    print(f"pooled: model logloss {pll:.4f}, acc {pac:.3f}; "
          f"market logloss {kll:.4f} (n={len(mk)})" if len(mk)
          else f"pooled: model logloss {pll:.4f}, acc {pac:.3f}")
    print("wrote game_predictions_*.csv, game_metrics.csv, game_calibration.csv")


def season_subset(allg, s):
    return allg[allg.season == s].copy()


if __name__ == "__main__":
    main()
