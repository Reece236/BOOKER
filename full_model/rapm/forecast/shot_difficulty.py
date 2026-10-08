"""
Defender-aware shot difficulty (the missing half of "shot quality").

The xFG model in shot_quality judges difficulty from the SHOT alone (distance, zone,
action type) -- it is blind to WHO was defending. A pull-up three against a top-5
defense counts the same as one against a bottom-5 defense, so tough-shot takers on
heavy diets (Booker-types) grade as "low 3P%" when they're really beating harder
conditions. True defender *proximity* is tracking-only data (not public); the best
available proxy -- used here -- is the ON-COURT DEFENDERS themselves:

  per shot: the 5 defenders (from shot_defense's shot->stint alignment) contribute
    - mean/max height (static, leak-free)
    - mean perimeter-contest + max rim-contest skill from the PRIOR season (leak-free)

A defender-aware xFG is refit per season with those features; each player's making
is then scored against what a league-average shooter hits GIVEN his shot types AND
the defenders he faced:

  d3p_adj    : difficulty-adjusted 3P% = league 3P% + (actual - expected | defenders)
  defg_adj   : difficulty-adjusted eFG% (same construction, all shots)
  dpts_oe100 : points over defender-aware expectation per 100 shots
  diet_xfg   : mean defender-aware expected make prob of his attempts (LOW = hard diet)

Output cache/shot_difficulty.csv per (season, PLAYER_ID), seasons 2018-2025 (2026 has
no public pbp; export carries the latest season forward, standard pattern).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import shot_quality as sq
from . import shot_defense as sdf

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
OUT = CACHE / "shot_difficulty.csv"
MIN_SHOTS = 50


def _height_map():
    h = pd.read_csv(CACHE / "player_heights.csv")
    hc = "height_in" if "height_in" in h.columns else "heightIn"
    return dict(zip(h.PLAYER_ID.astype(int), pd.to_numeric(h[hc], errors="coerce")))


def _prior_contest(season):
    """{pid: (perim_contest, rim_contest)} from the PRIOR season (leak-free)."""
    p = CACHE / "shot_defense.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[d.season == season - 1]
    return {int(r.PLAYER_ID): (float(r.perim_contest), float(r.rim_contest))
            for r in d.itertuples()}


def _def_features(merged, season, hmap):
    """Per-shot defender aggregates from the DEF_LINEUP string."""
    pc = _prior_contest(season)
    med_h = float(np.nanmedian(list(hmap.values())))
    hs_mean, hs_max, pc_mean, rc_max = [], [], [], []
    for lu in merged.DEF_LINEUP:
        pids = [int(x) for x in str(lu).split(",") if x.strip().isdigit()]
        hs = [hmap.get(p, med_h) for p in pids]
        hs = [h if h == h else med_h for h in hs]
        cons = [pc.get(p) for p in pids]
        perims = [c[0] for c in cons if c is not None]
        rims = [c[1] for c in cons if c is not None]
        hs_mean.append(np.mean(hs) if hs else med_h)
        hs_max.append(np.max(hs) if hs else med_h)
        pc_mean.append(np.mean(perims) if perims else 0.0)
        rc_max.append(np.max(rims) if rims else 0.0)
    return (np.array(hs_mean), np.array(hs_max), np.array(pc_mean), np.array(rc_max))


def _fit_dadj_xfg(df, seed=7):
    """Defender-aware xFG. Returns (probs, holdout_logloss_pair) where the pair is
    (defender-free, defender-aware) on the same 15% random holdout -- the gate."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    base_cols = pd.concat([
        df[["dist", "is3"]],
        pd.get_dummies(df.bucket, prefix="b"),
        pd.get_dummies(df.zone, prefix="z"),
    ], axis=1)
    dcols = df[["d_hmean", "d_hmax", "d_pcmean", "d_rcmax"]].reset_index(drop=True)
    Xb = base_cols.reset_index(drop=True)
    Xd = pd.concat([Xb, dcols], axis=1)
    y = df.made.values
    rng = np.random.default_rng(seed)
    hold = rng.random(len(df)) < 0.15
    mk = lambda: HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.06,
        min_samples_leaf=200, l2_regularization=1.0, random_state=7)
    mb = mk().fit(Xb.values[~hold], y[~hold])
    md = mk().fit(Xd.values[~hold], y[~hold])
    llb = log_loss(y[hold], mb.predict_proba(Xb.values[hold])[:, 1])
    lld = log_loss(y[hold], md.predict_proba(Xd.values[hold])[:, 1])
    # final probs from a full-data fit (holdout only gates the feature set)
    probs = mk().fit(Xd.values, y).predict_proba(Xd.values)[:, 1]
    return probs, (llb, lld)


def build(seasons=range(2018, 2027)):
    hmap = _height_map()
    rows = []
    for s in seasons:
        merged, _ = sdf._shots_with_defenders(s)
        if merged is None or not len(merged):
            # no public pbp for this season (e.g. 2026): DIET-ONLY fallback from the
            # shot chart alone -- defender features were a holdout wash anyway (the
            # adjustment's value is the shot-type/distance diet, which this keeps).
            sd_path = CACHE / f"shotdetail_{s}.csv"
            if not sd_path.exists():
                print(f"  {s}: no pbp and no shotdetail, skipped")
                continue
            merged = sq._featurize(pd.read_csv(sd_path, low_memory=False)).reset_index(drop=True)
            merged["dxfg"] = sq._xfg_model(merged)
            llb = lld = float("nan")
            print(f"  {s}: diet-only (no pbp for defender join)")
        else:
            merged = merged.reset_index(drop=True)
            hm, hx, pcm, rcx = _def_features(merged, s, hmap)
            merged["d_hmean"], merged["d_hmax"] = hm, hx
            merged["d_pcmean"], merged["d_rcmax"] = pcm, rcx
            probs, (llb, lld) = _fit_dadj_xfg(merged)
            merged["dxfg"] = probs
        lg3 = float(merged.loc[merged.is3 == 1, "made"].mean())
        lge = float((merged.made * (1 + 0.5 * merged.is3)).sum() / len(merged))
        g = merged.groupby("PLAYER_ID")
        agg = pd.DataFrame({
            "shots": g.size(),
            "made3": g.apply(lambda x: float((x.made * x.is3).sum())),
            "att3": g.apply(lambda x: float(x.is3.sum())),
            "x3": g.apply(lambda x: float((x.dxfg * x.is3).sum())),
            "efg_num": g.apply(lambda x: float((x.made * (1 + 0.5 * x.is3)).sum())),
            "xefg_num": g.apply(lambda x: float((x.dxfg * (1 + 0.5 * x.is3)).sum())),
            "pts": g.apply(lambda x: float((x.made * (2 + x.is3)).sum())),
            "xpts": g.apply(lambda x: float((x.dxfg * (2 + x.is3)).sum())),
            "diet_xfg": g.dxfg.mean(),
        }).reset_index()
        agg = agg[agg.shots >= MIN_SHOTS]
        agg["season"] = s
        # difficulty-adjusted rates: league base + (actual - expected), shrunk by volume
        agg["d3p_adj"] = np.where(agg.att3 >= 40,
                                  lg3 + (agg.made3 - agg.x3) / agg.att3, np.nan)
        agg["defg_adj"] = lge / max(1e-9, 1.0) + (agg.efg_num - agg.xefg_num) / agg.shots
        agg["dpts_oe100"] = 100.0 * (agg.pts - agg.xpts) / agg.shots
        rows.append(agg[["season", "PLAYER_ID", "shots", "att3", "d3p_adj",
                         "defg_adj", "dpts_oe100", "diet_xfg"]])
        gate = ("" if lld != lld else
                f" | xFG holdout logloss def-free {llb:.5f} -> def-aware {lld:.5f} "
                f"({'BETTER' if lld < llb else 'worse'})")
        print(f"  {s}: {len(merged)} shots{gate} | players {len(agg)}")
    out = pd.concat(rows, ignore_index=True)
    # merge-don't-clobber: keep previously built seasons not in this run
    if OUT.exists():
        old = pd.read_csv(OUT)
        old = old[~old.season.isin(set(out.season))]
        out = pd.concat([old, out], ignore_index=True).sort_values(["season", "PLAYER_ID"])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} player-seasons)")
    return out


if __name__ == "__main__":
    build()
