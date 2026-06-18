"""Out-of-sample backtest for BookerFormer vs ridge RAPM.

For each target season we train on the prior 3 seasons (no target-season leakage)
and evaluate three things the plan calls for:

  1. Team metrics: roster-aggregated predicted net rating vs actual team net
     (RMSE), and the net->wins map vs actual wins (RMSE / MAE / R2). Reuses
     player_impacts.aggregate_net + fit_net_to_wins so BookerFormer and ridge are
     scored on an identical pipeline.
  2. Held-out stint prediction: predict each target-season stint's offense
     points/100 and report possession-weighted RMSE. This is where the transformer's
     synergy/matchup capacity should beat additive ridge if anywhere.
  3. Uncertainty calibration (BookerFormer only): 50/80/90% interval coverage on
     held-out stints, and whether per-player sd shrinks with minutes.

Compared models:
  * ridge        -- build_impacts_off_def (the current linear O/D RAPM baseline)
  * former-add   -- BookerFormer with attention OFF (the RAPM-faithful sanity model)
  * former-attn  -- BookerFormer with attention ON (the proposed model)

Usage:
    python bookerformer_backtest.py                 # default seasons
    python bookerformer_backtest.py 2023 2024 2025  # specific targets
    python bookerformer_backtest.py --quick 2025    # fewer MC samples, faster
"""
import sys
from pathlib import Path

import numpy as np

from forecast import player_impacts as pi
from forecast import bookerformer as bf
from stint_off_def import enrich_season

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DEFAULT_TARGETS = [2022, 2023, 2024, 2025]
LEAGUE = pi.LEAGUE_PPP100


def _wmetrics(pred, actual, w):
    """Possession-weighted RMSE of stint offense-point predictions."""
    pred, actual, w = np.asarray(pred), np.asarray(actual), np.asarray(w)
    rmse = np.sqrt(np.average((pred - actual) ** 2, weights=w))
    return float(rmse)


def team_scores(data, impact, target, train):
    """Predicted-vs-actual team net RMSE and wins RMSE/MAE/R2 for an impact dict."""
    pred_net = pi.aggregate_net(data, impact, target)
    act_net = dict(zip(data.TEAMS[target].TEAM_ID, data.TEAMS[target].ACTUAL_NET))
    tids = [t for t in pred_net if t in act_net]
    pn = np.array([pred_net[t] for t in tids])
    an = np.array([act_net[t] for t in tids])
    net_rmse = float(np.sqrt(np.mean((pn - an) ** 2)))

    k, c = pi.fit_net_to_wins(data, train)
    pw, aw = [], []
    for t in tids:
        ab = data.abbr_of.get(t)
        if (ab, target) in data.ACTUAL_WINS:
            pw.append(k * pred_net[t] + c)
            aw.append(data.ACTUAL_WINS[(ab, target)])
    pw, aw = np.array(pw), np.array(aw)
    wins_rmse = float(np.sqrt(np.mean((pw - aw) ** 2)))
    wins_mae = float(np.mean(np.abs(pw - aw)))
    ss_res = np.sum((aw - pw) ** 2)
    ss_tot = np.sum((aw - aw.mean()) ** 2)
    wins_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return net_rmse, wins_rmse, wins_mae, wins_r2


def ridge_stint_pred(off, def_, home_lineups, away_lineups):
    """Ridge O/D prediction of offense points/100 for matchups (O scores vs D)."""
    preds, keep = [], []
    for ol, dl in zip(home_lineups, away_lineups):
        oids = [int(p) for p in ol]
        dids = [int(p) for p in dl]
        if len(oids) != 5 or len(dids) != 5:
            keep.append(False); continue
        keep.append(True)
        p = LEAGUE + sum(off.get(o, 0.0) for o in oids) - sum(def_.get(d, 0.0) for d in dids)
        preds.append(p)
    return np.array(preds), np.array(keep, dtype=bool)


def eval_season(data, target, quick=False):
    train = [s for s in range(target - 3, target) if s in data.STINTS]
    if len(train) < 2:
        return None
    mc = 15 if quick else bf.MC_SAMPLES
    rows = {}

    # held-out target stints (both matchup orientations) for stint-level scoring
    d = data.STINTS[target]
    home, away = list(d.home), list(d.away)
    poss = d.POSS.to_numpy()
    yoff = d.Y_OFF_HOME.to_numpy()           # home offense points/100
    ydef = d.Y_DEF_HOME.to_numpy()           # away offense points/100

    # ---- ridge O/D baseline -------------------------------------------------
    total, off, def_, _, _ = pi.build_impacts_off_def(data, train, target)
    nr, wr, wm, r2 = team_scores(data, total, target, train)
    p_off, keep = ridge_stint_pred(off, def_, home, away)
    rmse_off = _wmetrics(p_off, yoff[keep], poss[keep])
    rows["ridge"] = dict(net_rmse=nr, wins_rmse=wr, wins_mae=wm, wins_r2=r2,
                         stint_rmse=rmse_off, cov90=np.nan, sd_min_corr=np.nan,
                         ridge_corr=1.0)
    ridge_total = total                              # for the sanity-gate correlation

    # ---- BookerFormer (additive and attention) -----------------------------
    for label, attn in [("former-add", False), ("former-attn", True)]:
        post, model = bf.fit_bookerformer(train, target, data,
                                          use_attention=attn, mc_samples=mc)
        impact = dict(zip(post.PLAYER_ID, post.impact_total))
        nr, wr, wm, r2 = team_scores(data, impact, target, train)

        # sanity gate: per-player agreement with ridge RAPM (>0.95 expected for add)
        common = [p for p in impact if p in ridge_total]
        rc = float(np.corrcoef([impact[p] for p in common],
                               [ridge_total[p] for p in common])[0, 1])

        mu, sd, keepf = bf.predict_offense_points(model, model._fd, home, away,
                                                  mc_samples=mc)
        act = yoff[keepf]; w = poss[keepf]
        stint_rmse = _wmetrics(mu, act, w)
        # weighted 90% coverage (Gaussian predictive interval)
        z = 1.645
        inside = (act >= mu - z * sd) & (act <= mu + z * sd)
        cov90 = float(np.average(inside.astype(float), weights=w))
        # sd-vs-minutes (negative = narrows with minutes)
        pl = data.PLAYERS[target][["PLAYER_ID", "MINUTES"]]
        pp = post.merge(pl, on="PLAYER_ID", how="inner")
        pp = pp[pp.MINUTES >= 250]
        sdmc = float(np.corrcoef(pp.MINUTES, pp.sd_off)[0, 1]) if len(pp) > 5 else np.nan
        rows[label] = dict(net_rmse=nr, wins_rmse=wr, wins_mae=wm, wins_r2=r2,
                           stint_rmse=stint_rmse, cov90=cov90, sd_min_corr=sdmc,
                           ridge_corr=rc)
    return rows


def main(targets, quick=False):
    for s in range(2015, 2028):
        if (CACHE / f"stints_{s}.csv").exists():
            enrich_season(s)
    data = pi.BookerData(seasons=range(2015, 2028))

    agg = {}
    for target in targets:
        print(f"\n##### TARGET {target} #####")
        res = eval_season(data, target, quick=quick)
        if res is None:
            print("  (insufficient training seasons)")
            continue
        hdr = f"{'model':12s} {'net_rmse':>9s} {'wins_rmse':>9s} {'wins_mae':>8s} " \
              f"{'wins_r2':>8s} {'stint_rmse':>10s} {'cov90':>7s} {'sd~min':>7s} {'ridge~':>7s}"
        print(hdr)
        for label, m in res.items():
            print(f"{label:12s} {m['net_rmse']:9.3f} {m['wins_rmse']:9.2f} "
                  f"{m['wins_mae']:8.2f} {m['wins_r2']:8.3f} {m['stint_rmse']:10.2f} "
                  f"{m['cov90']:7.3f} {m['sd_min_corr']:7.3f} {m['ridge_corr']:7.3f}")
            agg.setdefault(label, []).append(m)

    print("\n##### AVERAGE ACROSS TARGETS #####")
    for label, ms in agg.items():
        f = lambda key: np.nanmean([m[key] for m in ms])
        print(f"{label:12s} net_rmse={f('net_rmse'):.3f}  wins_rmse={f('wins_rmse'):.2f}  "
              f"wins_r2={f('wins_r2'):.3f}  stint_rmse={f('stint_rmse'):.2f}  "
              f"cov90={f('cov90'):.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    quick = "--quick" in args
    args = [a for a in args if a != "--quick"]
    targets = [int(a) for a in args] or DEFAULT_TARGETS
    main(targets, quick=quick)
