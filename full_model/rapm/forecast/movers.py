"""
Progression / regression candidates, with the signal validated walk-forward.

Candidate score = BOOKER's projected change: PROJ(s+1) - rating(s). PROJ (forecast/
booker_projection.py) is trained only on pairs before s, so on past seasons this is a
genuine out-of-sample forecast of change. Drivers shown alongside (all known at the end of s):
  age           aging curve position
  form          second-half rating - full-season rating (h2_rapm)
  onoff_gap     real on-court +/- per 100 - isolated rating (lineup_context)
  luck3         raw 3P% - shrunk true 3P% on his volume (shot_quality) -- tested below for
                value BEYOND PROJ before being used as anything but a descriptive flag
Validation (players >= 1000 min in s and s+1, seasons with an observed s+1):
  * Spearman(predicted change, realized change) for realized = BookerFormer rating change
    and = next-season pure RAPM - rating (an arbiter that shares no prior with the model)
  * top / bottom 25 per season: share who actually improved / declined vs the base rate
  * partial test: does luck3 explain realized change after controlling for predicted change?
P(improve) for the current candidates = logistic fit of realized improvement on predicted
change over the validation seasons.

Writes cache/movers.csv (current season -> next) and cache/movers_validation.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import player_impacts as pi

CACHE = pi.CACHE
RAPM = CACHE.parent
MIN_MIN = 1000


def panel():
    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    proj = pd.read_csv(CACHE / "booker_proj.csv").rename(columns={"proj_for_season": "nxt"})
    X = rat[["season", "PLAYER_ID", "player", "team", "minutes", "impact_total", "impact_off", "impact_def"]].copy()
    X["nxt"] = X.season + 1
    X = X.merge(proj, on=["nxt", "PLAYER_ID"], how="left")
    X["pred_delta"] = X.proj_impact - X.impact_total
    nx = rat[["season", "PLAYER_ID", "impact_total", "minutes"]].rename(
        columns={"season": "nxt", "impact_total": "next_rating", "minutes": "next_min"})
    X = X.merge(nx, on=["nxt", "PLAYER_ID"], how="left")
    pr = pd.read_csv(CACHE / "pure_rapm.csv").rename(columns={"season": "nxt", "pure_rapm": "next_pure"})
    X = X.merge(pr[["nxt", "PLAYER_ID", "next_pure"]], on=["nxt", "PLAYER_ID"], how="left")
    h2 = pd.read_csv(CACHE / "h2_rapm.csv")
    X = X.merge(h2, on=["season", "PLAYER_ID"], how="left")
    X["form"] = X.h2_impact - X.impact_total
    lc = pd.read_csv(CACHE / "lineup_context.csv")[["season", "PLAYER_ID", "real_pm"]]
    X = X.merge(lc, on=["season", "PLAYER_ID"], how="left")
    X["onoff_gap"] = X.real_pm - X.impact_total
    sq = pd.read_csv(CACHE / "shot_quality.csv")[["season", "PLAYER_ID", "n3", "raw_3p", "true_3p"]]
    X = X.merge(sq, on=["season", "PLAYER_ID"], how="left")
    X["luck3"] = np.where(X.n3 >= 100, X.raw_3p - X.true_3p, np.nan)
    from . import uncertainty as unc
    X["age"] = unc.attach_attrs(X[["player", "season"]].assign(minutes=0)).age.values
    X["real_delta"] = X.next_rating - X.impact_total
    X["pure_delta"] = X.next_pure - X.impact_total
    return X


def validate(X):
    V = X[(X.minutes >= MIN_MIN) & (X.next_min >= MIN_MIN) & X.pred_delta.notna() & X.real_delta.notna()]
    out = {"n": int(len(V)), "seasons": sorted(int(s) for s in V.season.unique()), "by_season": []}
    hits = []
    for s, g in V.groupby("season"):
        top, bot = g.nlargest(25, "pred_delta"), g.nsmallest(25, "pred_delta")
        row = {"season": int(s), "n": int(len(g)),
               "rho_rating": round(float(spearmanr(g.pred_delta, g.real_delta)[0]), 3),
               "rho_pure": round(float(spearmanr(g.pred_delta, g.pure_delta, nan_policy="omit")[0]), 3),
               "top_improved": round(float((top.real_delta > 0).mean()), 3),
               "bottom_declined": round(float((bot.real_delta < 0).mean()), 3),
               "base_improved": round(float((g.real_delta > 0).mean()), 3),
               "top_mean_change": round(float(top.real_delta.mean()), 2),
               "bottom_mean_change": round(float(bot.real_delta.mean()), 2)}
        out["by_season"].append(row)
        hits.append(row)
    H = pd.DataFrame(hits)
    for k in ("rho_rating", "rho_pure", "top_improved", "bottom_declined", "base_improved",
              "top_mean_change", "bottom_mean_change"):
        out[k] = round(float(np.average(H[k], weights=H.n)), 3)
    # deciles of predicted change -> realized change (calibration of the change forecast)
    V = V.assign(dec=pd.qcut(V.pred_delta, 10, labels=False))
    out["deciles"] = [{"dec": int(d) + 1, "pred": round(float(g.pred_delta.mean()), 2),
                       "real": round(float(g.real_delta.mean()), 2),
                       "pure": round(float(g.pure_delta.mean()), 2),
                       "improved": round(float((g.real_delta > 0).mean()), 3)} for d, g in V.groupby("dec")]
    # does 3P luck add anything beyond the projection? (OLS of realized change)
    L = V.dropna(subset=["luck3"])
    A = np.c_[np.ones(len(L)), L.pred_delta, L.luck3]
    for tgt in ("real_delta", "pure_delta"):
        Lt = L.dropna(subset=[tgt])
        At = np.c_[np.ones(len(Lt)), Lt.pred_delta, Lt.luck3]
        b, *_ = np.linalg.lstsq(At, Lt[tgt].values, rcond=None)
        r = Lt[tgt].values - At @ b
        cov = np.linalg.inv(At.T @ At) * (r @ r) / (len(Lt) - 3)
        out[f"luck3_{tgt}"] = {"coef_per_10pct": round(float(b[2] * 0.1), 3),
                               "t": round(float(b[2] / np.sqrt(cov[2, 2])), 2), "n": int(len(Lt)),
                               "coef_pred": round(float(b[1]), 3)}
    # by starting tier: is the change signal just regression of extreme ratings?
    V = V.assign(tier=pd.cut(V.impact_total, [-99, -1, 1, 3, 99], labels=["< -1", "-1..1", "1..3", "> 3"]))
    out["by_tier"] = [{"tier": str(k), "n": int(len(g)),
                       "rho": round(float(spearmanr(g.pred_delta, g.real_delta)[0]), 3),
                       "pred": round(float(g.pred_delta.mean()), 2), "real": round(float(g.real_delta.mean()), 2)}
                      for k, g in V.groupby("tier", observed=True)]
    # calibration: P(improve) ~ logistic(pred_delta); expected change = isotonic(pred_delta)
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    lr = LogisticRegression(C=1e6).fit(V[["pred_delta"]], (V.real_delta > 0).astype(int))
    iso = IsotonicRegression(out_of_bounds="clip").fit(V.pred_delta, V.real_delta)
    out["p_improve_coef"] = [round(float(lr.intercept_[0]), 4), round(float(lr.coef_[0][0]), 4)]
    return out, (lr, iso)


def build():
    X = panel()
    val, (lr, iso) = validate(X)
    s = int(X[X.proj_impact.notna()].season.max())
    cur = X[(X.season == s) & X.proj_impact.notna()].copy()
    cur["p_improve"] = lr.predict_proba(cur[["pred_delta"]])[:, 1].round(3)
    cur["exp_delta"] = iso.predict(cur.pred_delta.values).round(2)     # calibrated to realized changes
    cols = ["season", "PLAYER_ID", "player", "team", "age", "minutes", "impact_total", "impact_off", "impact_def",
            "proj_impact", "pred_delta", "exp_delta", "p_improve", "form", "onoff_gap", "raw_3p", "true_3p", "luck3", "n3"]
    cur[cols].round(3).to_csv(CACHE / "movers.csv", index=False)
    json.dump(val, open(CACHE / "movers_validation.json", "w"), indent=1)
    print(f"movers: {len(cur)} candidates for {s + 1}; validation n={val['n']} over {val['seasons']}")
    print(f"  Spearman(pred change, realized): rating {val['rho_rating']}, pure RAPM {val['rho_pure']}")
    print(f"  top-25 improved {val['top_improved']:.0%} / bottom-25 declined {val['bottom_declined']:.0%} "
          f"(base improve {val['base_improved']:.0%}); mean realized change top {val['top_mean_change']:+.2f}, "
          f"bottom {val['bottom_mean_change']:+.2f}")
    print("  deciles pred->real:", [(d['pred'], d['real']) for d in val['deciles']])
    print("  3P luck beyond PROJ:", val["luck3_real_delta"], val["luck3_pure_delta"])
    print("  by starting tier:", val["by_tier"])
    return cur, val


if __name__ == "__main__":
    build()
