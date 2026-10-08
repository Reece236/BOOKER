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


def _eb_shrink(oe, svar, qual):
    """Normal-normal empirical-Bayes posterior mean (the TRUE-SKILL step).

    Given each player's observed make-over-expected `oe` and its sampling variance
    `svar`, shrink toward the league mean by the reliability of the sample. `qual`
    marks players with enough volume to estimate the population skill spread tau^2
    (method of moments: tau^2 = Var(oe) - mean sampling var). A full-season shooter
    barely moves; a 40-shot sample collapses most of the way to league average -- so
    the leaderboard ranks talent, not small-sample noise.
    """
    oe = np.asarray(oe, dtype=float)
    svar = np.asarray(svar, dtype=float)
    q = np.asarray(qual, dtype=bool)
    if q.sum() < 8:
        q = np.isfinite(svar)
    mu = float(np.mean(oe[q]))                       # league mean (~0 for over-expected)
    tau2 = max(0.0, float(np.var(oe[q]) - np.mean(svar[q])))
    shrink = tau2 / (tau2 + np.where(np.isfinite(svar), svar, np.inf))
    return mu + shrink * (oe - mu)


TS_DECAY = 0.70     # trailing-season weight: true skill = stable talent, not one season
TS_WINDOW = 3       # seasons of history folded into the talent estimate
# qualifying weighted-shot counts to estimate each category's population spread (tau^2)
QUAL = {"3": 140, "rim": 90, "efg": 220, "pts": 220}


def _summarize_season(df, s, min_shots):
    """Per-player-season RAW shot sufficient-stats + descriptive single-season columns.
    No shrinkage here -- the TRUE-SKILL step pools across seasons in `_true_skill_panel`.
    Each per-shot make variance uses xfg as p (xfg*(1-xfg))."""
    df = df.copy()
    df["ptval"] = df.is3 * 3 + (1 - df.is3) * 2
    df["efgval"] = df.is3 * 1.5 + (1 - df.is3) * 1.0
    df["vmake"] = df.xfg * (1 - df.xfg)
    df["rim"] = (df.zone == "Restricted Area").astype(int)
    df["selfc"] = df.bucket.isin(SELF_CREATE).astype(int)
    df["mpts"], df["xpts"] = df.made * df.ptval, df.xfg * df.ptval
    df["mefg"], df["xefg"] = df.made * df.efgval, df.xfg * df.efgval
    df["vpts"], df["vefg"] = df.ptval ** 2 * df.vmake, df.efgval ** 2 * df.vmake

    g = df.groupby("PLAYER_ID")
    M = g.agg(player=("PLAYER_NAME", "first"), shots=("made", "size"),
              made=("made", "sum"), sumx=("xfg", "sum"), xfg_mean=("xfg", "mean"),
              mpts=("mpts", "sum"), xpts=("xpts", "sum"), vpts=("vpts", "sum"),
              mefg=("mefg", "sum"), xefg=("xefg", "sum"), vefg=("vefg", "sum"),
              selfc=("selfc", "mean"), rim_rate=("rim", "mean"), three_rate=("is3", "mean"))
    for sub, pre in ((df[df.is3 == 1], "3"), (df[df.rim == 1], "_rim")):
        a = sub.groupby("PLAYER_ID").agg(**{
            f"n{pre}": ("made", "size"), f"made{pre}": ("made", "sum"),
            f"sumx{pre}": ("xfg", "sum"), f"v{pre}": ("vmake", "sum")})
        M = M.join(a, how="left")
    M = M[M.shots >= min_shots].copy()
    M = M.fillna({c: 0 for c in M.columns if c not in ("player", "shots")})

    base3 = float(df.loc[df.is3 == 1, "xfg"].mean()) if (df.is3 == 1).any() else 0.355
    base_rim = float(df.loc[df.rim == 1, "xfg"].mean()) if (df.rim == 1).any() else 0.62
    base_efg = float(df.xefg.sum() / max(len(df), 1))
    ns = M.shots.astype(float).values
    n3 = M.n3.astype(float).values
    M = M.reset_index().rename(columns={"PLAYER_ID": "PLAYER_ID"})
    M["season"] = int(s)
    M["base3"], M["base_rim"], M["base_efg"] = base3, base_rim, base_efg
    # descriptive single-season columns (raw_* are the actual season rates -> hover compare)
    M["xfg"] = M.xfg_mean.round(4)
    M["fg_oe"] = ((M.made - M.sumx) / ns).round(4)
    M["pts_oe100"] = ((M.mpts - M.xpts) / ns * 100).round(2)
    M["x3p"] = np.where(n3 > 0, M.sumx3 / np.maximum(n3, 1), np.nan).round(4)
    M["fg3_oe"] = np.where(n3 > 0, (M.made3 - M.sumx3) / np.maximum(n3, 1), np.nan).round(4)
    M["raw_3p"] = np.where(n3 > 0, M.made3 / np.maximum(n3, 1), np.nan).round(4)
    nr = M.n_rim.astype(float).values
    M["raw_rim"] = np.where(nr > 0, M.made_rim / np.maximum(nr, 1), np.nan).round(4)
    M["raw_efg"] = (M.mefg / ns).round(4)
    M["self_create"] = M.selfc.round(4)
    M["rim_rate"] = M.rim_rate.round(4)
    M["three_rate"] = M.three_rate.round(4)
    return M


def _true_skill_panel(raw):
    """Pool each player's sufficient-stats over a trailing `TS_WINDOW`-season decayed
    window, then empirical-Bayes shrink ACROSS players within the target season to get
    stable, difficulty-adjusted true-skill estimates. Returns the descriptive frame with
    true_3p / true_rim / true_efg / shot_making added."""
    SUM = ["n3", "made3", "sumx3", "v3", "n_rim", "made_rim", "sumx_rim", "v_rim",
           "shots", "mpts", "xpts", "vpts", "mefg", "xefg", "vefg"]
    by_pid = {pid: g.set_index("season") for pid, g in raw.groupby("PLAYER_ID")}
    acc = []
    for pid, gi in by_pid.items():
        for S in gi.index:
            w = {ss: TS_DECAY ** (S - ss) for ss in gi.index if 0 <= S - ss < TS_WINDOW}
            tot = {c: float(sum(w[ss] * gi.loc[ss, c] for ss in w)) for c in SUM if c not in ("v3", "v_rim", "vpts", "vefg")}
            # variances accumulate with weight^2 (independent shots)
            for c in ("v3", "v_rim", "vpts", "vefg"):
                tot[c] = float(sum(w[ss] ** 2 * gi.loc[ss, c] for ss in w))
            tot["PLAYER_ID"], tot["season"] = pid, S
            acc.append(tot)
    P = pd.DataFrame(acc)

    def cat(num_made, num_x, den_n, var, qual):
        den = P[den_n].values
        with np.errstate(invalid="ignore", divide="ignore"):
            oe = np.where(den > 0, (P[num_made].values - P[num_x].values) / den, 0.0)
            sv = np.where(den > 0, P[var].values / den ** 2, np.inf)
        return oe, sv, den >= qual

    out = {}
    for S, idx in P.groupby("season").groups.items():
        sl = P.loc[idx]
        b = raw[raw.season == S].iloc[0]
        for key, (nm, nx, dn, vr, q, base, scale, off) in {
            "true_3p":  ("made3", "sumx3", "n3", "v3", QUAL["3"], b.base3, 1.0, True),
            "true_rim": ("made_rim", "sumx_rim", "n_rim", "v_rim", QUAL["rim"], b.base_rim, 1.0, True),
            "true_efg": ("mefg", "xefg", "shots", "vefg", QUAL["efg"], b.base_efg, 1.0, True),
            "shot_making": ("mpts", "xpts", "shots", "vpts", QUAL["pts"], 0.0, 100.0, False),
        }.items():
            den = sl[dn].values
            with np.errstate(invalid="ignore", divide="ignore"):
                oe = np.where(den > 0, (sl[nm].values - sl[nx].values) / den, 0.0)
                sv = np.where(den > 0, sl[vr].values / den ** 2, np.inf)
            shrunk = _eb_shrink(oe, sv, den >= q)
            val = (base + shrunk) * scale if off else shrunk * scale
            for j, pid in enumerate(sl.PLAYER_ID.values):
                out.setdefault((pid, S), {})[key] = (
                    round(float(val[j]), 4 if off else 2) if den[j] > 0 else np.nan)

    raw = raw.copy()
    for key in ("true_3p", "true_rim", "true_efg", "shot_making"):
        raw[key] = [out.get((p, s), {}).get(key, np.nan)
                    for p, s in zip(raw.PLAYER_ID, raw.season)]
    keep = ["season", "PLAYER_ID", "player", "shots", "xfg", "fg_oe", "pts_oe100",
            "shot_making", "self_create", "rim_rate", "three_rate", "n3", "x3p", "fg3_oe",
            "raw_3p", "true_3p", "n_rim", "raw_rim", "true_rim", "raw_efg", "true_efg"]
    return raw[keep]


def player_shot_quality(seasons=range(2015, 2027), min_shots=50):
    frames = []
    for s in seasons:
        path = CACHE / f"shotdetail_{s}.csv"
        if not path.exists():
            continue
        df = _featurize(pd.read_csv(path, low_memory=False))
        if df.empty:
            continue
        df["xfg"] = _xfg_model(df)
        out_s = _summarize_season(df, s, min_shots)
        frames.append(out_s)
        print(f"  shot_quality {s}: {len(df)} shots, {len(out_s)} players")
    if not frames:
        return pd.DataFrame()
    return _true_skill_panel(pd.concat(frames, ignore_index=True))


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
    """{PLAYER_ID: composite offensive-skill z} for one season, z-scored vs qualified
    shooters. Two weightings (SQ_NUDGE_MODE env):
      legacy (default): half shot-making over expected, half self-created shot VOLUME.
      bible: OOS-validated -- shot QUALITY + SELECTION dominate (shot-making, rim-rate,
             three-rate) with self-creation VOLUME downweighted, per the Basketball
             Bible's "it starts on the shot" (volume != value). Final composite is
             standardized to unit SD so nudge_pts means the same in both modes."""
    import os
    if season in _SQ_CACHE:
        return _SQ_CACHE[season]
    out = {}
    if OUT.exists():
        df = pd.read_csv(OUT)
        df = df[(df.season == season) & (df.shots >= MIN_QUALIFY)].copy()
        if len(df) >= 20:
            def Z(s):
                s = s.astype(float)
                return (s - s.mean()) / (s.std() or 1.0)
            zmake = Z(df.pts_oe100)
            if os.environ.get("SQ_NUDGE_MODE", "bible") == "bible":
                comp = (0.43 * zmake + 0.35 * Z(df.rim_rate) + 0.18 * Z(df.three_rate)
                        + 0.10 * Z(df.self_create))
            else:
                comp = 0.5 * zmake + 0.5 * Z(df.self_create * df.shots)
            z = (comp - comp.mean()) / (comp.std() or 1.0)   # unit SD: comparable magnitude
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
