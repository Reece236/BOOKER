"""
Defensive shot value for BookerFormer.

Every opponent shot is tagged with the 5 DEFENDERS on the floor (offline lineup
reconstruction from nbastats play-by-play, reused from build_season_stints), then
graded two ways the eye test cares about:

  * deterrence    -- do opponents take WORSE shots when you're on? (lower expected
                     make `xfg` of allowed shots / fewer rim attempts)
  * make-limiting -- do opponents MAKE fewer than expected? (allowed `made - xfg` < 0)

Both fold into points-saved-over-expected per allowed shot, then a shot-level
offense/defense ridge (RAPM on the shot residual, controlling for the shooter)
de-collinearizes credit so a weak defender riding a great defensive unit doesn't
soak up value he didn't earn. Output cache/shot_defense.csv, per (season, PLAYER_ID):
  def_rapm      RAPM-adjusted points saved / 100 allowed shots (the headline)
  deterrence    league xfg - mean xfg allowed on-court (higher = forces worse shots)
  suppression   -(made - xfg) allowed on-court (higher = forces misses below expected)
  n_faced       allowed shots while on court

No defender-tracking data exists publicly, so "contest" is proxied by shot location
+ action type via the xfg model; this captures rim deterrence + shot-diet effects,
not literal closeout distance.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from . import shot_quality as sq

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
OUT = CACHE / "shot_defense.csv"
RIDGE_ALPHA = 1500.0


def _load_pbp(season):
    """nbastats play-by-play for a season (shuf year = season-1); download if absent."""
    import nba_on_court as noc
    shuf = season - 1
    tmp = CACHE / "_pbptmp"
    csv = CACHE / f"nbastats_{season}.csv"
    if not csv.exists():
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        noc.load_nba_data(path=tmp, seasons=[shuf], data="nbastats", untar=True)
        src = tmp / f"nbastats_{shuf}.csv"
        if not src.exists():
            shutil.rmtree(tmp)
            return None
        shutil.move(str(src), str(csv))
        shutil.rmtree(tmp)
    return pd.read_csv(csv, low_memory=False)


def _shots_with_defenders(season):
    """Return (shots_df, names) where each shot has its 5 defender PIDs + xfg + pts."""
    from build_season_stints import build_lineup_stints
    sd_path = CACHE / f"shotdetail_{season}.csv"
    pbp = _load_pbp(season)
    if pbp is None or not sd_path.exists():
        return None, {}
    pbp["GAME_ID"] = pd.to_numeric(pbp["GAME_ID"], errors="coerce").astype("Int64")

    # home team id per game (team credited with HOMEDESCRIPTION events)
    hd = pbp.dropna(subset=["HOMEDESCRIPTION"])
    home_team = hd.groupby("GAME_ID")["PLAYER1_TEAM_ID"].first().to_dict()

    # offline per-stint lineup windows
    st = build_lineup_stints(pbp)
    st = st[(st.HOME_LINEUP.str.count(",") == 4) & (st.AWAY_LINEUP.str.count(",") == 4)].copy()
    st["GAME_ID"] = pd.to_numeric(st.GAME_ID, errors="coerce")
    st["PERIOD"] = pd.to_numeric(st.PERIOD, errors="coerce")
    st = st.dropna(subset=["GAME_ID", "PERIOD", "START_SEC"])
    st["GAME_ID"] = st.GAME_ID.astype("int64"); st["PERIOD"] = st.PERIOD.astype("int64")
    st["START_SEC"] = st.START_SEC.astype("float64"); st["END_SEC"] = st.END_SEC.astype("float64")
    st = st.sort_values("START_SEC")

    # shots: xfg + clock
    shots = sq._featurize(pd.read_csv(sd_path, low_memory=False))
    shots["xfg"] = sq._xfg_model(shots)
    shots["GAME_ID"] = pd.to_numeric(shots["GAME_ID"], errors="coerce")
    shots["PERIOD"] = pd.to_numeric(shots["PERIOD"], errors="coerce")
    shots["sec"] = (pd.to_numeric(shots.MINUTES_REMAINING, errors="coerce") * 60
                    + pd.to_numeric(shots.SECONDS_REMAINING, errors="coerce")).astype("float64")
    shots = shots.dropna(subset=["GAME_ID", "PERIOD", "sec"])
    shots["GAME_ID"] = shots.GAME_ID.astype("int64"); shots["PERIOD"] = shots.PERIOD.astype("int64")
    shots = shots.sort_values("sec")

    # assign each shot to the active stint: first stint whose START_SEC >= shot sec
    merged = pd.merge_asof(
        shots, st[["GAME_ID", "PERIOD", "START_SEC", "END_SEC", "HOME_LINEUP", "AWAY_LINEUP"]],
        by=["GAME_ID", "PERIOD"], left_on="sec", right_on="START_SEC", direction="forward")
    merged = merged.dropna(subset=["HOME_LINEUP"])

    # defenders = lineup of the team NOT shooting
    def defenders(r):
        ht = home_team.get(r.GAME_ID)
        shooter_home = (ht is not None and r.TEAM_ID == ht)
        lu = r.AWAY_LINEUP if shooter_home else r.HOME_LINEUP
        return lu
    merged["DEF_LINEUP"] = merged.apply(defenders, axis=1)

    names = {}
    pj = CACHE / f"players_{season}.csv"
    if pj.exists():
        p = pd.read_csv(pj)
        names = dict(zip(p.PLAYER_ID.astype(int), p.NAME))
    return merged, names


def season_shot_defense(season):
    shots, names = _shots_with_defenders(season)
    if shots is None or shots.empty:
        return pd.DataFrame()
    val = (shots.is3 * 3 + (1 - shots.is3) * 2).astype(float)
    shots = shots.assign(pts=shots.made * val, xpts=shots.xfg * val)
    lg_pts = float(shots.pts.mean())
    shots["saved"] = lg_pts - shots.pts          # points saved vs league avg / shot
    shots["resid"] = shots.made - shots.xfg      # +made over expected (offense good)

    shots["is_rim"] = (shots.zone == "Restricted Area").astype(float)
    defs = shots.DEF_LINEUP.str.split(",").map(lambda xs: [int(x) for x in xs])
    shooters = shots.PLAYER_ID.astype(int).tolist()
    players = sorted(set(p for lu in defs for p in lu) | set(shooters))
    col = {p: i for i, p in enumerate(players)}
    nP = len(players)

    # shot-level off/def ridge: +1 for each of 5 defenders (def block), +1 for the
    # shooter (off block, offset nP). Fit TWO targets on the same design:
    #   points-saved  -> def_rapm   (overall: forces worse shots + forces misses)
    #   rim-attempt   -> rim_deter  (DETERRENCE: suppresses high-EV rim attempts)
    # Defender coef controls for the shooter, isolating the defender from teammates.
    rows, cols, vals = [], [], []
    for ri, (dl, sh) in enumerate(zip(defs, shooters)):
        for d in dl:
            rows.append(ri); cols.append(col[d]); vals.append(1.0)
        rows.append(ri); cols.append(nP + col[sh]); vals.append(1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(len(shots), 2 * nP))
    saved_coef = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(X, shots.saved.values).coef_[:nP]
    rim_coef = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(X, shots.is_rim.values).coef_[:nP]
    def_coef = dict(zip(players, saved_coef))
    # negative rim coef = defender lowers opponent rim-attempt rate = deterrence
    rimd = dict(zip(players, -rim_coef))

    # descriptive make-suppression (raw on-court make-over-expected allowed)
    expl = shots.assign(d=defs).explode("d")
    expl["d"] = expl.d.astype(int)
    agg = expl.groupby("d").agg(n_faced=("d", "size"), resid_allowed=("resid", "mean"))
    rows_out = []
    for pid, g in agg.iterrows():
        if g.n_faced < 200:
            continue
        rows_out.append({
            "season": int(season), "PLAYER_ID": int(pid),
            "player": names.get(int(pid), str(pid)),
            "n_faced": int(g.n_faced),
            "def_rapm": round(float(def_coef.get(pid, 0.0)) * 100, 2),   # pts saved /100 shots
            "rim_deter": round(float(rimd.get(pid, 0.0)) * 100, 2),      # +suppresses rim attempts (pct pts)
            "suppression": round(-float(g.resid_allowed), 4),            # +forces misses
        })
    out = pd.DataFrame(rows_out)
    print(f"  shot_defense {season}: {len(shots)} shots tagged, {len(out)} defenders")
    return out


def build(seasons=range(2015, 2027)):
    parts = [season_shot_defense(s) for s in seasons]
    parts = [p for p in parts if not p.empty]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty:
        out.to_csv(OUT, index=False)
        print(f"wrote {OUT} ({len(out)} player-seasons)")
    return out


# --- defensive prior nudge (fed into the BookerFormer defensive box prior) ----
_SD_CACHE = {}


def _season_def_z(season):
    if season in _SD_CACHE:
        return _SD_CACHE[season]
    out = {}
    if OUT.exists():
        df = pd.read_csv(OUT)
        df = df[(df.season == season) & (df.n_faced >= 300)].copy()
        if len(df) >= 20:
            # Weight the collinearity-resistant signals: rim-attempt deterrence (the
            # high-EV-shot-limiting the eye test wants) + make-suppression. The diffuse
            # points-saved def_rapm is downweighted -- it inherits the same lineup
            # collinearity that over-credits vets on elite-defense teams.
            z = lambda s: (s - s.mean()) / (s.std() or 1.0)
            comp = 0.65 * z(df.rim_deter) + 0.25 * z(df.suppression) + 0.10 * z(df.def_rapm)
            out = dict(zip(df.PLAYER_ID.astype(int), comp))
    _SD_CACHE[season] = out
    return out


def def_prior_nudge(train_seasons, target_season, pids, nudge_pts=1.0, decay=0.70):
    """Per-player DEFENSIVE prior nudge (points/100): decayed z-composite of the
    shot-based defensive value over the training seasons, scaled by nudge_pts/SD."""
    pidset = set(int(p) for p in pids)
    acc, wsum = {}, {}
    for s in train_seasons:
        z = _season_def_z(s)
        w = decay ** (target_season - 1 - s)
        for pid, zz in z.items():
            if pid in pidset:
                acc[pid] = acc.get(pid, 0.0) + w * zz
                wsum[pid] = wsum.get(pid, 0.0) + w
    return {p: nudge_pts * acc[p] / wsum[p] for p in acc if wsum[p] > 0}


if __name__ == "__main__":
    build()
