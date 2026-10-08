"""
Player DNA — a *continuous* player representation fusing four families:
  1. Skills   (difficulty-adjusted true-skill: shooting, playmaking, defense, rebounding)
  2. Size     (height)
  3. Usage    (shot diet, self-creation, on/off-ball gravity)
  4. Context  (minutes-weighted skill profile of the TEAMMATES on the floor with him)

That ~27-dim vector is standardized and reduced with PCA. NO discrete archetype buckets --
the representation is continuous. From the embedding we derive:
  - scarcity     : local density in embedding space (rare/unicorn profiles score high)
  - comparables  : nearest other-player seasons ("players like X")
and we expose an interpretable, continuous **style fingerprint** (percentile position on
named style axes) so a player is described by *where he sits*, not which bucket he's in.

Output cache/player_dna.csv (one row per qualifying player-season, 2018-2025; export carries
the most recent DNA forward to seasons the skill caches don't cover yet, e.g. 2026).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
OUT = CACHE / "player_dna.csv"

MIN_MIN = 600
N_PCA = 10
K_SCAR = 12
N_COMP = 6

OWN = ["true_3p", "true_rim", "true_efg", "shot_making", "self_create", "three_rate",
       "rim_rate", "true_ast36", "true_create36", "true_fta36", "true_ft_pct",
       "true_tov_pct", "true_stl36", "true_blk36", "true_reb36", "on_gravity",
       "off_gravity", "space_threat", "def_rapm", "perim_contest", "rim_contest", "height"]
TMCTX = ["true_create36", "on_gravity", "three_rate", "off_gravity", "height", "true_blk36"]

# interpretable continuous style axes (percentile position on each = the "fingerprint")
AXES = {
    "Playmaking": ["true_create36", "true_ast36"],
    "Ball Dominance": ["on_gravity", "self_create"],
    "Shooting": ["three_rate", "off_gravity", "true_3p", "space_threat"],
    "Rim / Interior": ["rim_rate", "true_reb36", "true_blk36", "height"],
    "Perimeter D": ["def_rapm", "perim_contest", "true_stl36"],
    "Efficiency": ["true_efg", "shot_making", "true_rim"],
}


def _load_features():
    def L(f, cols):
        d = pd.read_csv(CACHE / f"{f}.csv")
        return d[["PLAYER_ID", "season"] + [c for c in cols if c in d.columns]]
    sq = L("shot_quality", ["true_3p", "true_rim", "true_efg", "shot_making",
                            "self_create", "three_rate", "rim_rate"])
    pb = L("pbp_skills", ["true_ast36", "true_create36", "true_fta36", "true_ft_pct",
                          "true_tov_pct", "true_stl36", "true_blk36", "true_reb36", "minutes"])
    gv = L("gravity", ["on_gravity", "off_gravity", "space_threat"])
    sd = L("shot_defense", ["def_rapm", "perim_contest", "rim_contest"])
    X = (sq.merge(pb, on=["PLAYER_ID", "season"])
           .merge(gv, on=["PLAYER_ID", "season"])
           .merge(sd, on=["PLAYER_ID", "season"], how="left"))
    h = pd.read_csv(CACHE / "player_heights.csv")
    hc = "height_in" if "height_in" in h.columns else "heightIn"
    X = X.merge(h[["PLAYER_ID", hc]].rename(columns={hc: "height"}), on="PLAYER_ID", how="left")
    return X


def build(seasons=range(2018, 2027)):
    from . import player_impacts as pi
    X = _load_features()
    for c in OWN:
        if c not in X.columns:
            X[c] = np.nan
        X[c] = X[c].fillna(X[c].median())

    # teammate context via stints (minutes-weighted avg of the 4 other on-court players)
    med = {c: float(X[c].median()) for c in TMCTX}
    skill = {(int(r.season), int(r.PLAYER_ID)): [getattr(r, c) for c in TMCTX]
             for r in X.itertuples()}
    fallback = np.array([med[c] for c in TMCTX])
    data = pi.BookerData(seasons=range(min(seasons), max(seasons) + 1))
    acc = {}
    for s in seasons:
        d = data.STINTS.get(s)
        if d is None:
            continue
        for h5, a5, poss in zip(d.home, d.away, d.POSS):
            if poss <= 0:
                continue
            for lineup in (h5, a5):
                vecs = [np.array(skill.get((s, int(p)), fallback)) for p in lineup]
                tot = np.sum(vecs, axis=0)
                for k, p in enumerate(lineup):
                    a = acc.setdefault((s, int(p)), [0.0] + [0.0] * len(TMCTX))
                    a[0] += poss
                    tmavg = (tot - vecs[k]) / 4.0
                    for j in range(len(TMCTX)):
                        a[1 + j] += tmavg[j] * poss
    tm = pd.DataFrame([{"season": s, "PLAYER_ID": pid,
                        **{f"tm_{TMCTX[j]}": a[1 + j] / a[0] for j in range(len(TMCTX))}}
                       for (s, pid), a in acc.items() if a[0] > 0])
    X = X.merge(tm, on=["season", "PLAYER_ID"], how="left")
    TMFE = [f"tm_{c}" for c in TMCTX]
    for c in TMFE:
        X[c] = X[c].fillna(X[c].median())

    X = X[X.minutes >= MIN_MIN].reset_index(drop=True)
    FE = OWN + TMFE
    Z = StandardScaler().fit_transform(X[FE])
    pca = PCA(n_components=N_PCA, random_state=0).fit(Z)
    E = pca.transform(Z)
    for i in range(N_PCA):
        X[f"pc{i+1}"] = E[:, i].round(3)

    # continuous style fingerprint: percentile on each named axis
    zt = (X[OWN] - X[OWN].mean()) / X[OWN].std().replace(0, 1)
    style_pct = {}
    for ax, cols in AXES.items():
        raw = zt[cols].mean(axis=1)
        style_pct[ax] = (raw.rank(pct=True) * 100).round().astype(int)
    X["style_fp"] = ["|".join(f"{ax}:{style_pct[ax][i]}" for ax in AXES) for i in range(len(X))]

    # scarcity: mean distance to K nearest in standardized embedding
    En = StandardScaler().fit_transform(E)
    nn = NearestNeighbors(n_neighbors=K_SCAR + 1).fit(En)
    dist, _ = nn.kneighbors(En)
    X["scarcity"] = dist[:, 1:].mean(1).round(3)
    X["scarcity_pct"] = (X["scarcity"].rank(pct=True) * 100).round().astype(int)

    # comparables: nearest OTHER-player seasons (cosine on embedding)
    bf = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")[["PLAYER_ID", "player"]]
    name = bf.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").player.to_dict()
    nnc = NearestNeighbors(n_neighbors=30, metric="cosine").fit(E)
    dc, ic = nnc.kneighbors(E)
    pid_arr, seas_arr = X.PLAYER_ID.values, X.season.values
    comps = []
    for i in range(len(X)):
        out, seen = [], {int(pid_arr[i])}
        for rank, j in enumerate(ic[i]):
            if j == i:
                continue
            pj = int(pid_arr[j])
            if pj in seen:
                continue
            seen.add(pj)
            sim = round(1.0 - float(dc[i][rank]), 3)
            out.append(f"{pj}:{name.get(pj, pj)}:{int(seas_arr[j])}:{sim}")
            if len(out) >= N_COMP:
                break
        comps.append("|".join(out))
    X["comparables"] = comps

    keep = ["PLAYER_ID", "season", "style_fp", "scarcity", "scarcity_pct",
            "comparables"] + [f"pc{i+1}" for i in range(N_PCA)]
    X[keep].to_csv(OUT, index=False)
    ev = pca.explained_variance_ratio_
    print(f"wrote {OUT} ({len(X)} player-seasons, {X.season.min()}-{X.season.max()})")
    print(f"PCA var (top{N_PCA}): {[round(v,2) for v in ev]} sum={ev.sum():.2f}")
    for i in range(4):
        load = pd.Series(pca.components_[i], index=FE).sort_values()
        print(f"PC{i+1}: + [{', '.join(load.tail(3).index[::-1])}]  - [{', '.join(load.head(3).index)}]")
    return X


if __name__ == "__main__":
    build()
