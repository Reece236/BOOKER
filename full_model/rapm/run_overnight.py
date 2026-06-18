"""
Overnight BookerFormer training + posterior-diagnostics run.

Trains thoroughly (env-cranked iterations) for BOTH the full transformer model
(use_attention=True) and the additive Bayesian-RAPM baseline, across all seasons,
and saves everything needed to evaluate the fit:

  cache/bookerformer_diag/<tag>/
    history_<season>.csv     -- per-epoch train_loss / val_nll / KL (convergence)
    posterior_<season>.csv   -- per-player O/D posterior mean + SD (the variational
                                posterior; this is the model's belief about each rating)
    calibration_<season>.csv -- held-out-stint interval coverage at 50/68/80/90/95%
    ratings_<tag>.csv        -- full per-player ratings (impact, sd, BOOKER, WAA)
  cache/bookerformer_diag/summary.json
                             -- per-(tag,season) OOS team net/wins + calibration,
                                final val_nll, best epoch; plus pooled rollups.

Thorough fit is controlled by env vars read in forecast.bookerformer:
    BF_EPOCHS (default here 600), BF_PATIENCE (50), BF_MC (200).
Launch detached, e.g.:
    nohup .venv_rapm/bin/python full_model/rapm/run_overnight.py > /tmp/bf_overnight.log 2>&1 &

Nothing here overwrites the production booker_bookerformer_ratings.csv or data.js;
review the diagnostics first, then promote a model deliberately.
"""
import os
# thorough defaults (overridable from the launch environment) -- must be set BEFORE
# importing forecast.bookerformer, which reads them at import time.
os.environ.setdefault("BF_EPOCHS", "600")
os.environ.setdefault("BF_PATIENCE", "50")
os.environ.setdefault("BF_MC", "200")

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import player_impacts as pi
from forecast import bookerformer as bf
from stint_off_def import enrich_season

HERE = Path(__file__).resolve().parent
DIAG = HERE / "cache" / "bookerformer_diag"

SEASONS = [int(x) for x in os.environ.get("BF_SEASONS", "").split(",") if x.strip()] \
    or list(range(2018, 2028))
MODELS = [("additive", False), ("attention", True)]
# Gaussian interval half-widths for coverage levels
ZLEVELS = {"50": 0.674, "68": 0.994, "80": 1.282, "90": 1.645, "95": 1.960}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_posterior(model, fd, path):
    """Per-player variational posterior: O/D mean + SD (pts/100) + BOOKER-scale SD."""
    import torch
    with torch.no_grad():
        ao = model.a_off_mean.cpu().numpy()
        ad = model.a_def_mean.cpu().numpy()
        so = np.exp(0.5 * model.a_off_logvar.cpu().numpy())
        sd = np.exp(0.5 * model.a_def_logvar.cpu().numpy())
    rows = [{"PLAYER_ID": int(p), "player": fd.names.get(int(p), str(p)),
             "post_off_mean": round(float(ao[i]), 4), "post_off_sd": round(float(so[i]), 4),
             "post_def_mean": round(float(ad[i]), 4), "post_def_sd": round(float(sd[i]), 4),
             "prior_off": round(float(fd.prior_off[i]), 4),
             "prior_def": round(float(fd.prior_def[i]), 4)}
            for i, p in enumerate(fd.pids)]
    pd.DataFrame(rows).to_csv(path, index=False)


def calibration(model, fd, data, season, mc):
    """Held-out-stint predictive calibration: coverage at each level + RMSE + NLL."""
    d = data.STINTS[season]
    mu, sd, keep = bf.predict_offense_points(model, fd, list(d.home), list(d.away), mc_samples=mc)
    if not keep.any():
        return None
    act = d.Y_OFF_HOME.to_numpy()[keep]
    w = d.POSS.to_numpy()[keep]
    sd = np.maximum(sd, 1e-6)
    wsum = w.sum()
    rmse = float(np.sqrt(np.average((mu - act) ** 2, weights=w)))
    nll = float(np.average(0.5 * (np.log(2 * np.pi * sd ** 2) + ((act - mu) / sd) ** 2), weights=w))
    rows = []
    for lvl, z in ZLEVELS.items():
        inside = (np.abs(act - mu) <= z * sd).astype(float)
        rows.append({"level": int(lvl), "nominal": int(lvl) / 100.0,
                     "empirical_coverage": round(float(np.average(inside, weights=w)), 4)})
    return {"rows": rows, "rmse": round(rmse, 3), "nll": round(nll, 4),
            "n_stints": int(keep.sum()), "mean_pred_sd": round(float(np.average(sd, weights=w)), 2)}


def team_oos(data, impact, season, train):
    """Predicted-vs-actual team net RMSE and wins RMSE / R2 (out of sample)."""
    pred = pi.aggregate_net(data, impact, season)
    act = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ACTUAL_NET))
    tids = [t for t in pred if t in act]
    if len(tids) < 5:
        return None
    pn = np.array([pred[t] for t in tids]); an = np.array([act[t] for t in tids])
    net_rmse = float(np.sqrt(np.mean((pn - an) ** 2)))
    k, c = pi.fit_net_to_wins(data, train)
    pw, aw = [], []
    for t in tids:
        ab = data.abbr_of.get(t)
        if (ab, season) in data.ACTUAL_WINS:
            pw.append(k * pred[t] + c); aw.append(data.ACTUAL_WINS[(ab, season)])
    pw, aw = np.array(pw), np.array(aw)
    if len(pw) < 5 or not np.isfinite(net_rmse):
        return None
    wins_rmse = float(np.sqrt(np.mean((pw - aw) ** 2)))
    ss = np.sum((aw - pw) ** 2); tot = np.sum((aw - aw.mean()) ** 2)
    return {"net_rmse": round(net_rmse, 3), "wins_rmse": round(wins_rmse, 2),
            "wins_r2": round(float(1 - ss / tot), 3) if tot > 0 else None}


def main():
    t0 = time.time()
    import torch
    torch.set_num_threads(os.cpu_count() or 8)
    log(f"config: BF_EPOCHS={os.environ['BF_EPOCHS']} BF_PATIENCE={os.environ['BF_PATIENCE']} "
        f"BF_MC={os.environ['BF_MC']} BF_BATCH={os.environ.get('BF_BATCH','4096')} "
        f"threads={torch.get_num_threads()}")
    DIAG.mkdir(parents=True, exist_ok=True)
    for s in SEASONS:
        if (HERE / "cache" / f"stints_{s}.csv").exists():
            enrich_season(s)
    data = pi.BookerData(seasons=range(2015, 2028))
    mc = int(os.environ["BF_MC"])
    summary = {"config": {k: os.environ[k] for k in ("BF_EPOCHS", "BF_PATIENCE", "BF_MC")},
               "models": {}}

    for tag, attn in MODELS:
        outdir = DIAG / tag
        outdir.mkdir(parents=True, exist_ok=True)
        log(f"===== MODEL '{tag}' (attention={attn}) =====")
        rating_rows, season_summ = [], {}
        for target in SEASONS:
            train = [s for s in range(target - 3, target) if s in data.STINTS]
            if len(train) < 2:
                continue
            try:
                ts = time.time()
                post, model = bf.fit_bookerformer(train, target, data, use_attention=attn,
                                                  mc_samples=mc, verbose=False)
                # convergence history
                pd.DataFrame(model._history).to_csv(outdir / f"history_{target}.csv", index=False)
                save_posterior(model, model._fd, outdir / f"posterior_{target}.csv")
                # calibration + team OOS need actual stints/results (skip projection
                # seasons like 2027 that have no played games yet)
                cal = calibration(model, model._fd, data, target, mc) if target in data.STINTS else None
                if cal:
                    pd.DataFrame(cal["rows"]).to_csv(outdir / f"calibration_{target}.csv", index=False)
                # ratings rows (mirror build_bookerformer_ratings)
                pl = data.PLAYERS.get(target)
                k, _ = pi.fit_net_to_wins(data, train)
                impact = dict(zip(post.PLAYER_ID, post.impact_total))
                # OOS needs a played season (actual net/wins); 2027 is projection-only
                oos = team_oos(data, impact, target, train) if target in data.STINTS else None
                if pl is not None:
                    tmin = pl.groupby("TEAM_ID").MINUTES.transform("sum")
                    pl = pl.copy(); pl["presence"] = pl.MINUTES / (tmin / 5.0)
                    for r in post.itertuples():
                        row = pl[pl.PLAYER_ID == r.PLAYER_ID]
                        if row.empty:
                            continue
                        pres = float(row.presence.iloc[0])
                        rating_rows.append({
                            "season": target, "PLAYER_ID": int(r.PLAYER_ID), "player": r.NAME,
                            "team": data.abbr_of.get(int(row.TEAM_ID.iloc[0]), "?"),
                            "minutes": int(row.MINUTES.iloc[0]),
                            "impact_off": round(r.impact_off, 2), "impact_def": round(r.impact_def, 2),
                            "impact_total": round(r.impact_total, 2),
                            "sd_off": round(r.sd_off, 3), "sd_def": round(r.sd_def, 3),
                            "waa_off": round(k * r.impact_off * pres, 2),
                            "waa_def": round(k * r.impact_def * pres, 2),
                            "waa_total": round(k * r.impact_total * pres, 2),
                            "booker_score": round(k * r.impact_total * (3000.0 / 8200.0), 2),
                        })
                season_summ[str(target)] = {
                    "val_nll": round(float(model._history[-1]["val_nll"]), 4) if model._history else None,
                    "best_epoch": int(getattr(model, "_best_epoch", -1)),
                    "epochs_run": len(model._history),
                    "oos": oos,
                    "calibration": {"cov90": next((r["empirical_coverage"] for r in cal["rows"]
                                                   if r["level"] == 90), None),
                                    "stint_rmse": cal["rmse"], "nll": cal["nll"]} if cal else None,
                }
                log(f"  {tag} {target}: {len(post)} players, val_nll="
                    f"{season_summ[str(target)]['val_nll']}, best_epoch={season_summ[str(target)]['best_epoch']}, "
                    f"oos={oos}, cov90={season_summ[str(target)]['calibration']['cov90'] if cal else None} "
                    f"({time.time()-ts:.0f}s)")
            except Exception:
                log(f"  ERROR fitting {tag} {target}:\n{traceback.format_exc()}")
        if rating_rows:
            df = pd.DataFrame(rating_rows)
            df = df[df.minutes >= 250].copy()
            df["rank"] = df.groupby("season").waa_total.rank(ascending=False, method="first").astype(int)
            df.to_csv(outdir / f"ratings_{tag}.csv", index=False)
            log(f"  wrote ratings_{tag}.csv ({len(df)} rows)")
        # pooled OOS rollup
        oos_list = [v["oos"] for v in season_summ.values() if v.get("oos")]
        cal_list = [v["calibration"] for v in season_summ.values() if v.get("calibration")]
        summary["models"][tag] = {
            "by_season": season_summ,
            "pooled": {
                "wins_r2": round(float(np.mean([o["wins_r2"] for o in oos_list if o["wins_r2"] is not None])), 3) if oos_list else None,
                "net_rmse": round(float(np.mean([o["net_rmse"] for o in oos_list])), 3) if oos_list else None,
                "wins_rmse": round(float(np.mean([o["wins_rmse"] for o in oos_list])), 2) if oos_list else None,
                "cov90": round(float(np.mean([c["cov90"] for c in cal_list if c["cov90"] is not None])), 3) if cal_list else None,
                "stint_rmse": round(float(np.mean([c["stint_rmse"] for c in cal_list])), 2) if cal_list else None,
            },
        }
        (DIAG / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"  pooled[{tag}]: {summary['models'][tag]['pooled']}")

    (DIAG / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE in {(time.time()-t0)/60:.1f} min. Diagnostics in {DIAG}")


if __name__ == "__main__":
    main()
