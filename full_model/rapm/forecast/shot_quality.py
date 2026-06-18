"""
Shot-quality model for BookerFormer.

Fits a league expected-make model (xFG) per season from shot location + type +
action, then summarizes each player-season by HOW HARD their shots are and HOW WELL
they make them above expectation. This is what separates a selective, wide-open
catch-and-shoot specialist (high make %, but easy shots -> low value-add) from a
high-volume creator who makes contested pull-ups (lower raw %, but huge value-add).

Source: cache/shotdetail_<season>.csv (bulk shufinskiy shotdetail; see
data_ingest/fetch_shotdetail.py). One row per shot, with SHOT_DISTANCE, SHOT_ZONE_*,
SHOT_TYPE (2PT/3PT) and ACTION_TYPE (Pullup / Step Back / Jump Shot / Driving Layup
...). ACTION_TYPE is our (imperfect) proxy for self-creation / contest -- we have no
defender-distance/tracking data.

Per player-season outputs (cache/shot_quality.csv):
  shots          attempts modeled
  xfg            mean expected make prob of shots TAKEN  (LOW = harder shot diet)
  fg_oe          (made - expected) per shot              (shot-MAKING over expected)
  pts_oe100      points over expected per 100 shots      (value of the shot-making)
  self_create    share of FGA that are pull-up/step-back/fadeaway/turnaround
  rim_rate       share at the rim;  three_rate share from 3
  x3p / fg3_oe   expected 3P% of their 3s / 3P make over expected
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
OUT = CACHE / "shot_quality.csv"

# ACTION_TYPE -> coarse bucket. Order matters (first match wins).
_BUCKET_RULES = [
    ("stepback", r"step back"),
    ("pullup", r"pull[- ]?up"),
    ("fadeaway", r"fadeaway|turnaround"),
    ("floating", r"float"),
    ("dunk", r"dunk|alley"),
    ("putback", r"putback|tip"),
    ("hook", r"hook"),
    ("driving", r"driving|running|cutting|finger roll|reverse"),
    ("layup", r"layup"),
    ("jump", r"jump shot"),
]
# self-created / contested buckets (proxy for a harder shot the player made for himself)
SELF_CREATE = {"pullup", "stepback", "fadeaway"}


def _bucket(action):
    a = str(action).lower()
    for name, pat in _BUCKET_RULES:
        if re.search(pat, a):
            return name
    return "other"


def _featurize(df):
    df = df.copy()
    df["made"] = pd.to_numeric(df.SHOT_MADE_FLAG, errors="coerce")
    df = df[df.made.isin([0, 1])].copy()
    df["dist"] = pd.to_numeric(df.SHOT_DISTANCE, errors="coerce").clip(0, 35).fillna(0)
    df["is3"] = (df.SHOT_TYPE.astype(str) == "3PT Field Goal").astype(int)
    df["bucket"] = df.ACTION_TYPE.map(_bucket)
    df["zone"] = df.SHOT_ZONE_BASIC.astype(str)
    return df


def _xfg_model(df):
    """Per-season league expected-make model. Returns predicted make prob per shot."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    feat = pd.concat([
        df[["dist", "is3"]],
        pd.get_dummies(df.bucket, prefix="b"),
        pd.get_dummies(df.zone, prefix="z"),
    ], axis=1)
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.06,
        min_samples_leaf=200, l2_regularization=1.0, random_state=7)
    model.fit(feat.values, df.made.values)
    return model.predict_proba(feat.values)[:, 1]


def player_shot_quality(seasons=range(2015, 2027), min_shots=50):
    rows = []
    for s in seasons:
        path = CACHE / f"shotdetail_{s}.csv"
        if not path.exists():
            continue
        df = _featurize(pd.read_csv(path, low_memory=False))
        if df.empty:
            continue
        df["xfg"] = _xfg_model(df)
        df["pts"] = df.made * (df.is3 * 3 + (1 - df.is3) * 2)
        df["xpts"] = df.xfg * (df.is3 * 3 + (1 - df.is3) * 2)
        df["selfc"] = df.bucket.isin(SELF_CREATE).astype(int)
        df["rim"] = (df.zone == "Restricted Area").astype(int)
        for pid, g in df.groupby("PLAYER_ID"):
            n = len(g)
            if n < min_shots:
                continue
            g3 = g[g.is3 == 1]
            rows.append({
                "season": int(s), "PLAYER_ID": int(pid),
                "player": g.PLAYER_NAME.iloc[0], "shots": int(n),
                "xfg": round(float(g.xfg.mean()), 4),
                "fg_oe": round(float((g.made - g.xfg).mean()), 4),
                "pts_oe100": round(float((g.pts - g.xpts).mean() * 100), 2),
                "self_create": round(float(g.selfc.mean()), 4),
                "rim_rate": round(float(g.rim.mean()), 4),
                "three_rate": round(float(g.is3.mean()), 4),
                "n3": int(len(g3)),
                "x3p": round(float(g3.xfg.mean()), 4) if len(g3) else None,
                "fg3_oe": round(float((g3.made - g3.xfg).mean()), 4) if len(g3) else None,
            })
        print(f"  shot_quality {s}: {len(df)} shots, {sum(1 for r in rows if r['season']==s)} players")
    out = pd.DataFrame(rows)
    return out


def build(seasons=range(2015, 2027)):
    out = player_shot_quality(seasons)
    if not out.empty:
        out.to_csv(OUT, index=False)
        print(f"wrote {OUT} ({len(out)} player-seasons)")
    return out


# --- offensive skill prior nudge (fed into the BookerFormer box prior) -------
_SQ_CACHE = {}
MIN_QUALIFY = 200   # shots to qualify for the season skill distribution


def _season_skill_z(season):
    """{PLAYER_ID: composite offensive-skill z} for one season: half shot-making
    over expected, half self-created shot volume (z-scored vs qualified shooters)."""
    if season in _SQ_CACHE:
        return _SQ_CACHE[season]
    out = {}
    if OUT.exists():
        df = pd.read_csv(OUT)
        df = df[(df.season == season) & (df.shots >= MIN_QUALIFY)].copy()
        if len(df) >= 20:
            make = df.pts_oe100.astype(float)
            create = (df.self_create.astype(float) * df.shots.astype(float))
            zmake = (make - make.mean()) / (make.std() or 1.0)
            zcre = (create - create.mean()) / (create.std() or 1.0)
            z = 0.5 * zmake + 0.5 * zcre
            out = dict(zip(df.PLAYER_ID.astype(int), z))
    _SQ_CACHE[season] = out
    return out


def skill_prior_nudge(train_seasons, target_season, pids, nudge_pts=1.0, decay=0.70):
    """Per-player OFFENSIVE prior nudge (points/100): decayed blend of the composite
    skill z over the training seasons, scaled by nudge_pts (per SD). Added to the
    offensive box prior so skilled high-usage creators start higher; the stint/RAPM
    likelihood still pulls the posterior back toward on-court results."""
    pidset = set(int(p) for p in pids)
    acc, wsum = {}, {}
    for s in train_seasons:
        z = _season_skill_z(s)
        w = decay ** (target_season - 1 - s)
        for pid, zz in z.items():
            if pid in pidset:
                acc[pid] = acc.get(pid, 0.0) + w * zz
                wsum[pid] = wsum.get(pid, 0.0) + w
    return {p: nudge_pts * acc[p] / wsum[p] for p in acc if wsum[p] > 0}


if __name__ == "__main__":
    build()
