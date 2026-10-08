"""Projected minutes SHARE per team-season -- no realized minutes anywhere.

Every team aggregate used to weight players by the minutes they ACTUALLY played in
the target season (and assign them to the team they ended the season on): an oracle
that knows injuries, trades and rotation changes. This module replaces that with a
projection made from information available before the season:

  roster   historical: players who appear for the team in its first OPEN_GAMES regular-
           season games (the opening rotation -- known within ~2 weeks; preseason
           depth charts carry the same information). Mid-season acquisitions are NOT
           on it, exactly as a preseason forecast wouldn't know them.
           live (2027): ESPN rosters in cache/players_{season}.csv.
  features (all from seasons <= s-1)
           prev / prev2 presence (minutes as a fraction of a full rotation slot),
           games-played share last season, rating THROUGH s-1 (ratings row s-1) and
           its rank / z within the new roster, prev-presence rank within the roster,
           FIT = number of roster teammates in the same position group rated above
           him (depth-chart competition), age, rookie, changed-team.
  model    ridge (+ prev x rating, prev^2, age^2) on realized presence on that team (walk-forward: season
           s is predicted by a model trained on seasons < s). Shares are NOT inflated to
           5: opening rosters historically absorb only ~4.5 of 5 presence (the rest goes
           to mid-season trades/signings/call-ups), so a team's projected presence is
           capped at 5 and the unallocated remainder is filled at REPLACEMENT -- the
           presence-weighted prior rating of players who joined after the opening games
           (estimated walk-forward from history; see team_nets).
  live     ESPN rosters run ~20 deep (two-ways, camp bodies); only the 15 players with
           the highest raw projection (the active-roster limit, matching the ~15-man
           historical opening rotations the model is trained on) receive minutes.

presence = player minutes on team / (team player-minutes / 5); a 36-mpg 82-game
starter is ~0.75. Output cache/minutes_proj.csv:
    season, team, PLAYER_ID, name, proj_pres, proj_minutes, rating_prev, src
Season 2027 also rewrites cache/players_2027.csv MINUTES (overrides respected).

Run:  python -m forecast.minutes_share
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import player_impacts as pi
from . import uncertainty as unc

CACHE = pi.CACHE
RAPM = CACHE.parent
OUT = CACHE / "minutes_proj.csv"
OPEN_GAMES = 10
SLOT_MIN = 82 * 48.0          # one rotation slot over a full season (presence 1.0)
FIRST_TRAIN = 2018
FEATS = ["prev", "prev2", "gp_prev", "rating", "rating_missing", "r_rank", "r_z",
         "p_rank", "pos_depth", "age", "rookie", "changed", "roster_n", "roster_prev_sum"]
POS_GROUP = {"PG": "G", "SG": "G", "SF": "W", "PF": "B", "C": "B"}


# --------------------------------------------------------------------------- data
def team_game_seconds(data, s):
    """Long frame: GAME_ID, DATE, team, PLAYER_ID, sec (regular season, 5v5 stints)."""
    d = data.STINTS.get(s)
    g = data.GAMES.get(s)
    if d is None or g is None:
        return None
    d = d[d.GAME_ID.astype(str).str.startswith("2")]
    g = g[g.SEASON_TYPE == "Regular Season"]
    home = dict(zip(g.GAME_ID.astype("int64"), g.HOME))
    away = dict(zip(g.GAME_ID.astype("int64"), g.AWAY))
    date = dict(zip(g.GAME_ID.astype("int64"), g.DATE))
    acc = {}
    for gid, hl, al, dur in zip(d.GAME_ID.astype("int64"), d.home, d.away, d.DURATION_SECONDS):
        for side, lu in ((home.get(gid), hl), (away.get(gid), al)):
            if side is None:
                continue
            for p in lu:
                k = (gid, side, p)
                acc[k] = acc.get(k, 0.0) + float(dur)
    out = pd.DataFrame([(k[0], date[k[0]], k[1], k[2], v) for k, v in acc.items()],
                       columns=["GAME_ID", "DATE", "team", "PLAYER_ID", "sec"])
    return out


def season_presence(tgs):
    """Realized presence per (team, player) and per-player league presence."""
    tsec = tgs.groupby("team").sec.sum()
    tp = tgs.groupby(["team", "PLAYER_ID"]).sec.sum().reset_index()
    tp["pres"] = tp.sec / tp.team.map(tsec / 5.0)
    slot = tsec.mean() / 5.0                               # league slot (handles 2020/21)
    pl = tgs.groupby("PLAYER_ID").agg(sec=("sec", "sum"), gp=("GAME_ID", "nunique"))
    team_games = tgs.groupby("team").GAME_ID.nunique().mean()
    pl["pres_l"] = pl.sec / slot
    pl["gp_frac"] = pl.gp / team_games
    main = tp.sort_values("sec").drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID").team
    pl["main_team"] = main
    return tp, pl


def opening_roster(tgs, n=OPEN_GAMES):
    """{team: [pids]} -- players appearing in the team's first n games."""
    g = tgs[["GAME_ID", "DATE", "team"]].drop_duplicates().sort_values(["team", "DATE", "GAME_ID"])
    g["k"] = g.groupby("team").cumcount()
    early = set(map(tuple, g[g.k < n][["GAME_ID", "team"]].values))
    e = tgs[[(gid, t) in early for gid, t in zip(tgs.GAME_ID, tgs.team)]]
    e = e.groupby(["PLAYER_ID", "team"]).sec.sum().reset_index()
    e = e.sort_values("sec").drop_duplicates("PLAYER_ID", keep="last")   # early-trade: keep bigger
    return e.groupby("team").PLAYER_ID.apply(list).to_dict()


# ------------------------------------------------------------------- features
class _Ctx:
    def __init__(self):
        self.data = pi.BookerData()
        self.rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
        self.tgs, self.tp, self.pl = {}, {}, {}
        for s in self.data.seasons:
            t = team_game_seconds(self.data, s)
            if t is not None and len(t):
                self.tgs[s] = t
                self.tp[s], self.pl[s] = season_presence(t)
        m = pd.read_csv(unc.MASTER, usecols=["playerName", "season", "position"])
        m["nm"] = m.playerName.map(pi.norm_name)
        m = m.dropna(subset=["position"]).sort_values("season")
        self.pos = {}
        for nm, s, p in zip(m.nm, m.season, m.position):
            self.pos.setdefault(nm, []).append((int(s), str(p).split("-")[0]))
        self.names = {}
        for s, p in self.data.PLAYERS.items():
            self.names.update(dict(zip(p.PLAYER_ID.astype(int), p.NAME)))

    def position(self, pid, s):
        lst = self.pos.get(pi.norm_name(self.names.get(pid, "")), [])
        prior = [p for ss, p in lst if ss < s] or [p for _, p in lst]
        return POS_GROUP.get(prior[-1], "W") if prior else "W"


def features(ctx, s, roster):
    """roster {team: [pid]} -> feature frame for season s (inputs strictly <= s-1)."""
    r1 = ctx.rat[ctx.rat.season == s - 1].set_index("PLAYER_ID").impact_total
    p1 = ctx.pl.get(s - 1)
    p2 = ctx.pl.get(s - 2)
    rows = []
    for team, pids in roster.items():
        for pid in pids:
            pid = int(pid)
            prev = float(p1.pres_l.get(pid, 0.0)) if p1 is not None else 0.0
            prev2 = float(p2.pres_l.get(pid, 0.0)) if p2 is not None else 0.0
            rows.append({
                "season": s, "team": team, "PLAYER_ID": pid,
                "name": ctx.names.get(pid, str(pid)),
                "prev": prev, "prev2": prev2,
                "gp_prev": float(p1.gp_frac.get(pid, 0.0)) if p1 is not None else 0.0,
                "rating": float(r1.get(pid, np.nan)),
                "changed": int(p1 is not None and pid in p1.index and p1.main_team.get(pid) != team),
                "rookie": int(prev == 0 and prev2 == 0),
                "pos": ctx.position(pid, s),
            })
    X = pd.DataFrame(rows)
    if X.empty:
        return X
    X["rating_missing"] = X.rating.isna().astype(int)
    X["rating"] = X.rating.fillna(pi.PRIOR_BASE)
    a = unc.attach_attrs(X.rename(columns={"name": "player"})[["player", "season"]].assign(minutes=0))
    X["age"] = a.age.values
    g = X.groupby("team")
    X["r_rank"] = g.rating.rank(ascending=False)
    X["r_z"] = (X.rating - g.rating.transform("mean")) / g.rating.transform("std").replace(0, 1)
    X["p_rank"] = g.prev.rank(ascending=False)
    X["roster_n"] = g.PLAYER_ID.transform("size")
    X["roster_prev_sum"] = g.prev.transform("sum")
    X["pos_depth"] = [int(((X.team == t) & (X.pos == p) & (X.rating > r)).sum())
                      for t, p, r in zip(X.team, X.pos, X.rating)]
    return X


def realized(ctx, s, X):
    tp = ctx.tp[s].set_index(["team", "PLAYER_ID"]).pres
    return np.array([float(tp.get((t, p), 0.0)) for t, p in zip(X.team, X.PLAYER_ID)])


ROSTER_MAX = 15


def normalize(X, col="raw"):
    """Calibrated raw shares, scaled DOWN only if a team's sum exceeds 5."""
    v = X[col].clip(lower=0.0, upper=0.85)
    tot = v.groupby(X.team).transform("sum")
    return v * np.minimum(1.0, 5.0 / tot.replace(0, 1))


def replacement_rating(ctx, seasons):
    """Presence-weighted prior rating (ratings row s-1; PRIOR_BASE if unrated) of
    players who played for a team in season s but were NOT on its opening roster."""
    num = den = 0.0
    for s in seasons:
        if s not in ctx.tgs:
            continue
        ros = opening_roster(ctx.tgs[s])
        r1 = ctx.rat[ctx.rat.season == s - 1].set_index("PLAYER_ID").impact_total
        tp = ctx.tp[s]
        for t, p, pres in zip(tp.team, tp.PLAYER_ID, tp.pres):
            if p in set(ros.get(t, [])):
                continue
            num += pres * float(r1.get(p, pi.PRIOR_BASE)); den += pres
    return num / den if den else pi.PRIOR_BASE


def _design(X):
    Z = X[FEATS].copy()
    Z["age"] = Z.age.fillna(27.0)
    Z["age2"] = (Z.age - 27.0) ** 2
    Z["prev_x_rz"] = X.prev * X.r_z          # good players keep their minutes
    Z["prev_sq"] = X.prev ** 2
    return Z


class _RidgeShare:
    """Standardized ridge with interactions. Chosen over gradient boosting on the
    walk-forward panel (2020-26): RMSE 587 vs 593 min, star calibration (prev >= .55:
    proj 1977 vs realized 1970 min; GBM 1934), within-team rank corr .623 vs .616 --
    and it extrapolates the tails honestly (trees compress stars)."""
    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(_design(X), y)
        return self

    def predict(self, X):
        return self.m.predict(_design(X))


def _model():
    return _RidgeShare()


# ------------------------------------------------------------------- build
def build(write_live=True, verbose=True):
    ctx = _Ctx()
    obs = sorted(s for s in ctx.tgs if s - 1 in ctx.pl and (ctx.rat.season == s - 1).any())
    panel = {}
    for s in obs:
        X = features(ctx, s, opening_roster(ctx.tgs[s]))
        X["y"] = realized(ctx, s, X)
        panel[s] = X
    outs, evals = [], []
    for s in obs:
        prior = [panel[t] for t in obs if FIRST_TRAIN <= t < s]
        tr = pd.concat(prior) if prior else None
        X = panel[s].copy()
        if tr is None or len(tr) < 400:
            X["raw"] = X.prev.where(X.prev > 0, 0.15)          # cold start: last year's share
            src = "naive"
        else:
            m = _model().fit(tr, tr.y)
            X["raw"] = m.predict(X)
            src = "model"
        X["proj_pres"] = normalize(X)
        X["naive"] = normalize(X.assign(raw=X.prev.where(X.prev > 0, 0.15)))
        X["src"] = src
        X["repl"] = replacement_rating(ctx, [t for t in obs if t < s]) if s > obs[0] else pi.PRIOR_BASE
        outs.append(X)
        if src == "model":
            mm = 3936.0     # presence -> minutes for an 82-game season
            evals.append({"season": s, "n": len(X),
                          "rmse_model_min": np.sqrt(np.mean((X.proj_pres - X.y) ** 2)) * mm,
                          "rmse_naive_min": np.sqrt(np.mean((X.naive - X.y) ** 2)) * mm})
    # live season(s): rosters from players_{s}.csv where no stints exist yet
    for s in sorted(ctx.data.PLAYERS):
        if s in ctx.tgs or s - 1 not in ctx.pl:
            continue
        pl = ctx.data.PLAYERS[s]
        tm = ctx.data.TEAMS.get(s)
        ab = dict(zip(tm.TEAM_ID.astype(int), tm.ABBR)) if tm is not None else ctx.data.abbr_of
        roster = {}
        for pid, tid in zip(pl.PLAYER_ID.astype(int), pl.TEAM_ID):
            if tid == tid and int(tid) in ab:
                roster.setdefault(ab[int(tid)], []).append(pid)
        X = features(ctx, s, roster)
        tr = pd.concat([panel[t] for t in obs if t >= FIRST_TRAIN])
        mdl = _model().fit(tr, tr.y)
        X["raw"] = mdl.predict(X)
        # Two-pass: preseason ESPN rosters carry camp invites (~21 deep vs ~15 in the
        # training rotations), which deflates the roster-relative features (roster_n,
        # ranks, z, depth, roster_prev_sum) and every share with them. Pick the likely
        # 15-man rotation, recompute those features on it, and re-predict.
        keep = X.groupby("team").raw.rank(ascending=False, method="first") <= ROSTER_MAX
        rot = {t: g.PLAYER_ID.tolist() for t, g in X[keep].groupby("team")}
        X2 = features(ctx, s, rot)
        X2["raw"] = mdl.predict(X2)
        rest = X[~keep].copy()
        rest["raw"] = 0.0
        X = pd.concat([X2, rest[X2.columns.intersection(rest.columns)]], ignore_index=True)
        X["proj_pres"] = normalize(X)
        X["repl"] = replacement_rating(ctx, obs)
        X = _apply_overrides(X, s)
        X["y"] = np.nan
        X["src"] = "model-live"
        outs.append(X)
        if write_live:
            _write_players(ctx, s, X)
    out = pd.concat(outs, ignore_index=True)
    out["proj_minutes"] = (out.proj_pres * SLOT_MIN).round(1)
    out["rating_prev"] = out.rating.round(2)
    out[["season", "team", "PLAYER_ID", "name", "proj_pres", "proj_minutes", "rating_prev",
         "src", "repl", "y"]].rename(columns={"y": "realized_pres"}).to_csv(OUT, index=False)
    if verbose and evals:
        E = pd.DataFrame(evals)
        print(E.round(1).to_string(index=False))
        print(f"pooled player RMSE (minutes, walk-forward): model {np.sqrt((E.rmse_model_min**2 * E.n).sum()/E.n.sum()):.0f}"
              f" vs last-season-share {np.sqrt((E.rmse_naive_min**2 * E.n).sum()/E.n.sum()):.0f}")
        print(f"wrote {OUT} ({len(out)} rows)")
    return out


def _apply_overrides(X, s):
    p = CACHE / f"minutes_overrides_{s}.csv"
    if not p.exists():
        return X
    ov = pd.read_csv(p)
    for r in ov.itertuples():
        m = X.name.map(pi.norm_name) == pi.norm_name(r.NAME)
        if not m.any():
            continue
        team = X.loc[m, "team"].iloc[0]
        fixed = float(r.MINUTES) / SLOT_MIN
        others = (X.team == team) & ~m
        delta = fixed - float(X.loc[m, "proj_pres"].iloc[0])
        X.loc[m, "proj_pres"] = fixed
        osum = X.loc[others, "proj_pres"].sum()      # teammates absorb the difference
        X.loc[others, "proj_pres"] *= max(0.0, osum - delta) / osum if osum else 1.0
        print(f"  override {r.NAME}: {r.MINUTES} min ({str(r.REASON)[:50]})")
    return X


def _write_players(ctx, s, X):
    p = CACHE / f"players_{s}.csv"
    pl = pd.read_csv(p)
    mins = dict(zip(X.PLAYER_ID, (X.proj_pres * SLOT_MIN).round(1)))
    pl["MINUTES"] = [mins.get(int(pid), m) for pid, m in zip(pl.PLAYER_ID, pl.MINUTES)]
    pl.to_csv(p, index=False)
    print(f"  players_{s}.csv MINUTES <- projected minutes share ({len(mins)} players)")


# ------------------------------------------------------------------- consumers
def load(season=None):
    df = pd.read_csv(OUT)
    return df if season is None else df[df.season == season]


def team_nets(impact, season, proj=None, age_shift=None, repl=None):
    """{team: sum(proj_pres * impact) + (5 - sum proj_pres) * replacement} over the
    projected roster. `impact` maps pid -> rating (prior-only!); unknown players get
    PRIOR_BASE. `age_shift` optional pid -> additive aging delta."""
    proj = load(season) if proj is None else proj
    v = proj.PLAYER_ID.map(lambda p: impact.get(int(p), pi.PRIOR_BASE))
    if age_shift:
        v = v + proj.PLAYER_ID.map(lambda p: age_shift.get(int(p), 0.0))
    rp = float(proj.repl.iloc[0]) if repl is None and "repl" in proj and len(proj) else \
        (pi.PRIOR_BASE if repl is None else repl)
    pres = proj.proj_pres.groupby(proj.team).sum()
    core = (v * proj.proj_pres).groupby(proj.team).sum()
    return (core + (5.0 - pres).clip(lower=0) * rp).to_dict()


def team_shares(season, proj=None):
    """{team: {pid: proj_pres}} for consumers that need the roster itself."""
    proj = load(season) if proj is None else proj
    return {t: dict(zip(g.PLAYER_ID.astype(int), g.proj_pres)) for t, g in proj.groupby("team")}


if __name__ == "__main__":
    build()
