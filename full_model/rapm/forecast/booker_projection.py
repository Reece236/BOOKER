"""
BOOKER-PROJ: next-season player-impact forecast (the "beat DARKO" model).

Walk-forward gradient boosting stacking the assets the eval proved are leading
indicators: current lineup-isolated impact (off/def), prior-season impact (memory),
difficulty-adjusted skills (shot-making, diet-adj 3P, shot selection, creation, FTA,
TOV, stocks/reb), spacing threat, gravity, age curve, rating uncertainty, minutes,
and SECOND-HALF-SEASON form (cache/h2_rapm.csv -- the recency signal that closed most
of the DARKO gap). Scored vs DARKO on neutral arbiters: statistical dead heat overall
(.361 vs .375, CI spans 0), WINS on team-changers (.332 vs .301).

Leak-free convention: the row with feature-season t produces the projection FOR season
t+1, trained only on (t', t'+1) pairs with feature season t' < t (outcomes <= t).

Output cache/booker_proj.csv: proj_for_season, PLAYER_ID, proj_impact.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from . import player_impacts as pi
from . import uncertainty as unc

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
OUT = CACHE / "booker_proj.csv"

FEATS = ["impact_off", "impact_def", "imp_prev", "shot_making", "three_rate", "rim_rate",
         "self_create", "d3p_adj", "diet_xfg", "true_create36", "true_fta36",
         "true_tov_pct", "true_stl36", "true_blk36", "true_reb36", "space_threat",
         "on_gravity", "age", "age2", "sd_tot", "usage", "minutes",
         "h2_impact", "form",
         # 2-yr memory (the Kalman-style long-memory DARKO advantage): 2-seasons-ago
         # impact, year-over-year impact/skill/minutes deltas. OOS: small consistent
         # gain, stacks with the debiased target; a 3rd year adds nothing.
         "imp_prev2", "d_imp", "d_three_rate", "d_shot_making", "min_prev"]


def _L(f, cols):
    p = CACHE / f
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[["PLAYER_ID", "season"] + [c for c in cols if c in d.columns]]


def _panel():
    bk = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    X = bk[["PLAYER_ID", "season", "player", "minutes",
            "impact_off", "impact_def", "impact_total", "sd_off", "sd_def"]].copy()
    prev = bk[["PLAYER_ID", "season", "impact_total"]].rename(columns={"impact_total": "imp_prev"})
    prev = prev.copy(); prev["season"] += 1
    X = X.merge(prev, on=["PLAYER_ID", "season"], how="left")
    prev2 = bk[["PLAYER_ID", "season", "impact_total"]].rename(columns={"impact_total": "imp_prev2"})
    prev2 = prev2.copy(); prev2["season"] += 2
    X = X.merge(prev2, on=["PLAYER_ID", "season"], how="left")
    pm = bk[["PLAYER_ID", "season", "minutes"]].rename(columns={"minutes": "min_prev"})
    pm = pm.copy(); pm["season"] += 1
    X = X.merge(pm, on=["PLAYER_ID", "season"], how="left")
    X["d_imp"] = X.impact_total - X.imp_prev
    for f, cols in (("shot_quality.csv", ["shot_making", "three_rate", "rim_rate", "self_create"]),
                    ("shot_difficulty.csv", ["d3p_adj", "diet_xfg"]),
                    ("pbp_skills.csv", ["true_create36", "true_fta36", "true_tov_pct",
                                        "true_stl36", "true_blk36", "true_reb36"]),
                    ("gravity.csv", ["space_threat", "on_gravity"]),
                    ("h2_rapm.csv", ["h2_impact"])):
        d = _L(f, cols)
        if d is not None:
            X = X.merge(d, on=["PLAYER_ID", "season"], how="left")
    X["form"] = X.h2_impact - X.impact_total
    for c in ("three_rate", "shot_making"):
        pv = X[["PLAYER_ID", "season", c]].rename(columns={c: c + "_pv"}).copy()
        pv["season"] += 1
        X = X.merge(pv, on=["PLAYER_ID", "season"], how="left")
        X["d_" + c] = X[c] - X[c + "_pv"]
    a = unc.attach_attrs(X[["player", "season"]].assign(minutes=X.minutes))
    X["age"] = a.age.values; X["usage"] = a.usage.values
    X["age2"] = (X.age - 27.0) ** 2
    X["sd_tot"] = np.hypot(X.sd_off, X.sd_def)
    nxt = bk[["PLAYER_ID", "season", "impact_total"]].rename(columns={"impact_total": "y_imp"})
    nxt = nxt.copy()
    # only OBSERVED seasons may be training targets: the ratings CSV also carries
    # future-season rows that are themselves projections (no stints exist) --
    # training on them would be circular.
    obs = {int(s) for s in nxt.season.unique() if (CACHE / f"stints_{int(s)}.csv").exists()}
    nxt = nxt[nxt.season.isin(obs)]
    nxt["season"] -= 1
    return X.merge(nxt, on=["PLAYER_ID", "season"], how="left")


def _lob_scores(X):
    """Play-finisher score per row: z(rim_rate) - z(three_rate) - z(self_create),
    z-pooled within season among >=750-min players, clipped +-2.5, floored at 0.
    High = Gafford/Capela-type lob finisher with no spacing or self-creation."""
    out = pd.Series(0.0, index=X.index)
    for _, g in X.groupby("season"):
        pool = g[g.minutes >= 750]
        if len(pool) < 100:
            continue
        def z(c):
            mu, sd = pool[c].mean(), pool[c].std()
            return (((g[c] - mu) / sd) if sd else g[c] * 0.0).clip(-2.5, 2.5)
        sc = (z("rim_rate") - z("three_rate") - z("self_create")).clip(lower=0)
        out.loc[g.index] = sc.fillna(0.0)
    return out


def build():
    # RIDGE, not gradient boosting: OOS accuracy is a statistical tie (.358 vs .361
    # Spearman) but trees compress the tails (rare extreme seasons regress to the
    # training mass -- Jokic +15.7 forecast +7.4, absurd), while the linear model
    # extrapolates honestly (+11.5). Tail fidelity matters for display.
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    X = _panel()
    X["lob_score"] = _lob_scores(X)
    # on/off gap: real on-court net minus the isolated rating. The 2026-07 bias
    # audit showed a calibrated dose of on-court reality predicts next season
    # beyond the isolated impact (resid corr +0.17 vs pure RAPM, both arbiters,
    # FDR-significant) -- some of what the lineup model regularizes away is real.
    lcp = CACHE / "lineup_context.csv"
    if lcp.exists():
        lc = pd.read_csv(lcp)[["PLAYER_ID", "season", "real_pm"]]
        X = X.merge(lc, on=["PLAYER_ID", "season"], how="left")
        X["onoff_gap"] = X.real_pm - X.impact_total
        lcn = lc.copy(); lcn["season"] -= 1
        X = X.merge(lcn.rename(columns={"real_pm": "y_real"}),
                    on=["PLAYER_ID", "season"], how="left")
    else:
        X["onoff_gap"] = np.nan; X["y_real"] = np.nan
    # next-season pure RAPM, keyed back to the feature season.
    prp = CACHE / "pure_rapm.csv"
    if prp.exists():
        arb = pd.read_csv(prp)[["PLAYER_ID", "season", "pure_rapm"]].copy()
        arb["season"] -= 1
        X = X.merge(arb.rename(columns={"pure_rapm": "y_pure"}),
                    on=["PLAYER_ID", "season"], how="left")
    else:
        X["y_pure"] = np.nan
    # Partial-feed guard: when a season's stint capture is incomplete (2026: ~72%
    # of expected stint-hours -- the feed died mid-season), its stint-derived
    # recency/on-court features are computed on a biased subset and read as
    # nonsense (Jokic 2026 "form" -15). Null them for EVERY row of a flagged
    # season (no cherry-picking); the ridge median-fill / recal z-fill handle
    # missingness exactly as for any player without the data.
    hours = {}
    for s in X.season.unique():
        p = CACHE / f"stints_{int(s)}.csv"
        if p.exists():
            hours[int(s)] = pd.read_csv(p, usecols=["DURATION_SECONDS"]).DURATION_SECONDS.sum() / 3600.0
    for s, h in hours.items():
        prior = [hours[t] for t in hours if t < s]
        if prior and h < 0.85 * max(prior):
            m = X.season == s
            X.loc[m, ["h2_impact", "form", "real_pm", "onoff_gap"]] = np.nan
            print(f"  partial-feed season {s}: stint hours {h:.0f} < 85% of prior max "
                  f"{max(prior):.0f} -> recency/on-court features nulled")

    # residual-calibration arbiter: 65/35 pure-RAPM / real +/- blend (std-aligned).
    # Pure-only optimized the isolated arbiter but left a real_pm deficit vs DARKO;
    # the 65/35 blend converts the pure surplus into fixing it (OOS: +.014 real_pm
    # at -.001 pure). The std ratio is a unit conversion, not signal.
    sl = X.y_pure.std() / (X.y_real.std() or 1.0)
    X["arb"] = np.where(X.y_pure.notna() & X.y_real.notna(),
                        0.65 * X.y_pure + 0.35 * X.y_real * sl,
                        np.where(X.y_pure.notna(), X.y_pure, X.y_real * sl))

    # Residual-calibration features (2026-07 bias audit, all independently
    # significant on BOTH neutral arbiters after FDR):
    #   lob_score   play finishers overrated (the ridge target inherits the
    #               lineup model's bias -- Gafford forecast +5.9 pre-fix)
    #   onoff_gap   on-court reality underweighted (z, clipped +-1.5: the tighter
    #               clip ties the +-2.5 gate on accuracy but tames juggernaut
    #               role-player tails; team-demeaning HURT, don't do it)
    #   form        2nd-half recency the ridge underweights vs its biased target
    #   true_stl36  defensive playmaking leads future value
    #   age         residual aging beyond the ridge's age/age2 terms
    # Fit arb ~ a + b*pred + sum g_i f_i on PRIOR seasons' walk-forward preds;
    # correction = sum (g_i/b) f_i. OOS gate: forward Spearman .316 -> .350
    # (pure) / .290 -> .345 (real_pm); onoff/form/steals/age resid corrs die;
    # leftover mover effect -0.05 (p=.09) = below the bar, no mover term.
    CFEATS = ["lob_score", "onoff_gap", "form", "true_stl36", "age"]
    for f in CFEATS[1:]:
        z = X.groupby("season")[f].transform(
            lambda v: (v - v.mean()) / (v.std() or 1.0))
        X["z_" + f] = z.clip(-1.5, 1.5).fillna(0.0)
    X["z_lob_score"] = X.lob_score.fillna(0.0)

    rows = []
    made = {}          # feature season -> (te frame, raw ridge preds, recal'd preds)
    for s in sorted(X.season.unique()):
        tr = X[(X.season < s) & X.y_imp.notna()].copy()
        te = X[X.season == s]
        if len(tr) < 400 or not len(te):
            continue
        # Debiased training target: 0.4*next impact + 0.3*next pure RAPM +
        # 0.3*next real +/- (std-aligned within the fold). Training on the raw
        # next BookerFormer impact re-learns that model's own biases; the blend
        # was the single biggest OOS lever vs DARKO (movers pure +.033).
        bp, br = tr.y_pure.notna(), tr.y_real.notna()
        sp_ = (tr.y_imp[bp].std() / (tr.y_pure[bp].std() or 1.0)) if bp.any() else 1.0
        sr_ = (tr.y_imp[br].std() / (tr.y_real[br].std() or 1.0)) if br.any() else 1.0
        tr["y_tgt"] = (0.4 * tr.y_imp
                       + 0.3 * np.where(bp, tr.y_pure * sp_, tr.y_imp)
                       + 0.3 * np.where(br, tr.y_real * sr_, tr.y_imp))
        med = tr[FEATS].median().fillna(0.0)   # early folds: imp_prev2 all-NaN
        sc = StandardScaler().fit(tr[FEATS].fillna(med))
        m = Ridge(alpha=10.0).fit(sc.transform(tr[FEATS].fillna(med)), tr.y_tgt)
        pred = m.predict(sc.transform(te[FEATS].fillna(med)))
        gam = np.zeros(len(CFEATS))
        tr_p, tr_f, tr_a = [], [], []
        for t in made:
            tf, tp, _ = made[t]
            ok = tf.arb.notna().to_numpy()
            tr_p.append(tp[ok])
            tr_f.append(tf[["z_" + f for f in CFEATS]].to_numpy()[ok])
            tr_a.append(tf.arb.to_numpy()[ok])
        if tr_p and sum(len(v) for v in tr_p) >= 400:
            P = np.concatenate(tr_p); F = np.vstack(tr_f); Y = np.concatenate(tr_a)
            A = np.column_stack([np.ones(len(P)), P, F])
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
            if coef[1] > 0.05:
                gam = coef[2:] / coef[1]
        adj = pred + te[["z_" + f for f in CFEATS]].to_numpy() @ gam
        # Remap to impact units (walk-forward y_imp ~ a + b*adj on prior folds;
        # rank-preserving, b ~ 1.35): the blended target compresses the scale,
        # and the display must read like impact per 100 (stars look like stars).
        hp_p, hp_y = [], []
        for t in made:
            tf, _, ta = made[t]
            ok = tf.y_imp.notna().to_numpy()
            hp_p.append(ta[ok]); hp_y.append(tf.y_imp.to_numpy()[ok])
        out_v = adj
        if hp_p and sum(len(v) for v in hp_p) >= 300:
            P = np.concatenate(hp_p); Y = np.concatenate(hp_y)
            (a_, b_), *_ = np.linalg.lstsq(np.column_stack([np.ones(len(P)), P]), Y,
                                           rcond=None)
            out_v = a_ + b_ * adj
        made[s] = (te, pred, adj)
        for pid, v in zip(te.PLAYER_ID, out_v):
            rows.append({"proj_for_season": int(s) + 1, "PLAYER_ID": int(pid),
                         "proj_impact": round(float(v), 2)})
        gtxt = " ".join(f"{f}={g:+.2f}" for f, g in zip(CFEATS, gam))
        print(f"  proj for {s+1}: {len(te)} players (trained on {len(tr)} pairs; {gtxt})")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    build()
