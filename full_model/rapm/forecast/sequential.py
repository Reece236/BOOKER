"""Sequential (game-by-game) Bayesian updating of player O/D ratings and minutes shares.

For each season s the filter starts from PRIOR-ONLY information and updates after every
game date using only games already played -- no realized minutes, no same-game data:

  prior     player (o, d) = BookerFormer rating THROUGH s-1 (ratings row s-1), aged one
            year; variance = rating sd^2 + ETA^2 (year-over-year innovation). Players with
            no row get a replacement-level prior with wide variance.
  state     theta = [o_1..o_P, d_1..d_P, mu, h]; mu = league scoring drift vs last
            season's level, h = home-offense edge (pts/100).
  update    Kalman filter per game DATE on every 5v5 stint of that date (two rows per
            stint: home offense vs away defense and vice versa), with REAL per-side
            points: y = pts/100 - L_{s-1} = sum(o_off) - sum(d_def) + mu + h*home + e,
            Var(e) = C / POSS (C estimated from season s-1 residual scatter). Between
            dates the ratings drift (random walk, Q per day) so the filter keeps learning.
  shares    team presence per player = recency-weighted blend of the preseason projection
            (forecast.minutes_share, weight K0 games) and observed per-game presence
            (0 when on the roster but DNP -> injuries pull shares down; trades move a
            player to his new team on his first appearance there). Unallocated presence
            (5 - sum) is filled at the replacement rating.
  pregame   team net = sum(share * (o + d)) + fill * repl; three availability modes:
              strict  : unconditional expected shares -- nothing about tonight's game
              pregame : P(plays tonight) from consecutive games just missed + role
                        (logistic, fit on prior seasons), expected presence via the
                        fitted redistribution -- only information public before tip
              actives : who actually suits up (injury-report upper bound), same
                        redistribution; still no minutes from the game

Outputs (per season):
  cache/seq_games_{s}.csv    GAME_ID, date, home, away, net_home/away (strict, actives)
  cache/seq_players_{s}.parquet  pre-game per player per game: o, d, sd, share, played
  cache/seq_team_daily_{s}.csv   date, team, net (strict) BEFORE that date's games
Run:  python -m forecast.sequential [seasons...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import minutes_share as ms
from . import player_impacts as pi
from . import uncertainty as unc

CACHE = pi.CACHE
RAPM = CACHE.parent

ETA = 1.0          # year-over-year innovation sd per side (pts/100)
NEW_SD = 2.0       # prior sd per side for players with no rating
Q_DAY = 0.0015     # per-day rating drift variance per side
MU_SD, Q_MU = 2.0, 0.01
H_MEAN, H_SD = 2.5, 1.0
K0 = 4.0           # projection weight in games for the share blend (tuned 2019-21: 4 > 8 > 16)
HL_GAMES = 10.0    # recency half-life of observed presence
AVAIL = 0.85       # typical availability: projected (unconditional) share / AVAIL = share when playing
# Minutes redistribution when players sit (availability_research.py, fit on realized
# per-game presence 2019-21, tested 2022-26): absent minutes go to available teammates
# with weight c^ALPHA (c = share when playing). ALPHA < 1 -> the deeper bench absorbs
# relatively more; a same-position term fit to zero. Test presence RMSE .1352 -> .1315.
ALPHA = 0.4


def avail_features(streak, c):
    """Features for P(plays tonight): consecutive team games just missed (one-hot 1..12,
    0 = played last, -1 = hasn't played for this team yet) and log share-when-playing."""
    s = np.clip(np.asarray(streak), -1, 12)
    cols = [(s == k).astype(float) for k in range(1, 13)]
    lc = np.log(np.clip(np.nan_to_num(np.asarray(c, float), nan=0.02), 0.01, None))
    return np.column_stack(cols + [(s < 0).astype(float), lc, lc * (s == 0)])


def fit_availability(panel, before_season):
    from sklearn.linear_model import LogisticRegression
    tr = panel[panel.season < before_season]
    if len(tr) < 20000:
        return None
    return LogisticRegression(C=10, max_iter=3000).fit(avail_features(tr.streak, tr.c), tr.played)


def redistribute(cond, avail):
    """Expected presence {pid: share} for a team-game given share-when-playing `cond`
    and availability (probability or 0/1) `avail`; sums to 5."""
    pids = list(cond)
    c = np.array([max(cond[p], 1e-3) for p in pids]); a = np.array([avail.get(p, 0.0) for p in pids])
    M = float(np.sum((1 - a) * c))
    w = a * c ** ALPHA
    pres = a * c + (M * w / w.sum() if w.sum() > 0 else 0.0)
    tot = pres.sum()
    return dict(zip(pids, pres * (5.0 / tot))) if tot > 0 else {}


def _league(d):
    return 100.0 * float(d.HOME_PTS.sum() + d.AWAY_PTS.sum()) / (2.0 * float(d.POSS.sum()))


def _noise_c(d, L):
    """Per-possession variance of a side's points (x100^2), from last season's stints.
    Pooled ratio estimator sum((pts - ppp*poss)^2) / sum(poss): the naive mean of
    poss*(y-L)^2 explodes on sub-second stints (2022 gave 71k vs ~19k elsewhere)."""
    ppp = L / 100.0
    pts = np.concatenate([d.HOME_PTS, d.AWAY_PTS]).astype(float)
    poss = np.concatenate([d.POSS, d.POSS]).astype(float)
    ok = np.isfinite(pts) & (poss > 0)
    return float(1e4 * np.sum((pts[ok] - ppp * poss[ok]) ** 2) / np.sum(poss[ok]))


def run_season(data, s, rat, proj, repl, verbose=True, avail_model=None, injury_log=None):
    reg_games = data.GAMES[s]
    reg_games = reg_games[reg_games.SEASON_TYPE == "Regular Season"].sort_values(["DATE", "GAME_ID"])
    st = data.STINTS.get(s)
    st = st[st.GAME_ID.astype(str).str.startswith("2")] if st is not None else None
    prev = data.STINTS[s - 1]
    L = _league(prev)
    C = _noise_c(prev, L)

    from . import injury_availability as ia
    name_to_pid = {}
    for ss_, pl_ in data.PLAYERS.items():
        name_to_pid.update({pi.norm_name(n_): int(p_) for p_, n_ in zip(pl_.PLAYER_ID, pl_.NAME)})
    # ---- player universe + priors -------------------------------------------------
    r1 = rat[rat.season == s - 1].set_index("PLAYER_ID")
    a = unc.attach_attrs(r1.reset_index()[["player", "season"]].assign(minutes=0))
    ages = dict(zip(r1.index, a.age.values))
    pids = set(proj.PLAYER_ID.astype(int))
    if st is not None:
        for hl, al in zip(st.home, st.away):
            pids.update(hl); pids.update(al)
    pids = sorted(pids)
    P = len(pids)
    ix = {p: i for i, p in enumerate(pids)}
    D = 2 * P + 2
    m = np.zeros(D)
    v = np.zeros(D)
    for p, i in ix.items():
        if p in r1.index:
            o, dd = float(r1.at[p, "impact_off"]), float(r1.at[p, "impact_def"])
            ag = ages.get(p)
            if ag == ag and ag is not None:            # one year of aging, split O/D
                da = pi.AGE_QUAD * ((ag + 1 - pi.AGE_PEAK) ** 2 - (ag - pi.AGE_PEAK) ** 2)
                o += da / 2; dd += da / 2
            m[i], m[P + i] = o, dd
            v[i] = float(r1.at[p, "sd_off"]) ** 2 + ETA ** 2
            v[P + i] = float(r1.at[p, "sd_def"]) ** 2 + ETA ** 2
        else:
            m[i] = m[P + i] = repl / 2.0
            v[i] = v[P + i] = NEW_SD ** 2
    m[2 * P], v[2 * P] = 0.0, MU_SD ** 2
    m[2 * P + 1], v[2 * P + 1] = H_MEAN, H_SD ** 2
    S = np.diag(v)
    qd = np.full(D, Q_DAY); qd[2 * P] = Q_MU; qd[2 * P + 1] = 0.0

    # ---- per-game presence + team roster bookkeeping --------------------------------
    tgs = ms.team_game_seconds(data, s)
    gpres = {}
    if tgs is not None and len(tgs):
        tsec = tgs.groupby(["GAME_ID", "team"]).sec.transform("sum")
        tgs = tgs.assign(pres=tgs.sec / (tsec / 5.0))
        for (gid, team), g in tgs.groupby(["GAME_ID", "team"]):
            gpres[(int(gid), team)] = dict(zip(g.PLAYER_ID.astype(int), g.pres))
    proj_sh = {t: dict(zip(g.PLAYER_ID.astype(int), g.proj_pres)) for t, g in proj.groupby("team")}
    hist = {t: [] for t in proj_sh}                     # team -> list of per-game dicts
    member = {}                                         # pid -> current team
    for t, d_ in proj_sh.items():
        for p in d_:
            member[p] = t
    stints_by_date = ({dt: g for dt, g in st.assign(DATE=st.GAME_ID.astype("int64").map(
        dict(zip(reg_games.GAME_ID.astype("int64"), reg_games.DATE)))).groupby("DATE")}
        if st is not None else {})

    def shares(team, conditional=False):
        """Unconditional expected presence (DNPs count as 0) or, if `conditional`, the
        expected presence GIVEN he plays (for the actives mode: tonight's availability
        is known, so his injury-depressed share must not be double-counted)."""
        base = proj_sh.get(team, {})
        h = hist.get(team, [])
        cand = {p for p in base if member.get(p) == team}
        for g in h:
            cand |= {p for p in g if member.get(p) == team}
        n = len(h)
        w = 0.5 ** ((n - 1 - np.arange(n)) / HL_GAMES) if n else np.zeros(0)
        out = {}
        for p in cand:
            if conditional:
                ws = [(wk, g[p]) for wk, g in zip(w, h) if g.get(p, 0.0) > 0]
                obs = sum(wk * x for wk, x in ws); wt = sum(wk for wk, _ in ws)
                out[p] = (K0 * base.get(p, 0.0) / AVAIL + obs) / (K0 + wt)
            else:
                obs = sum(wk * g.get(p, 0.0) for wk, g in zip(w, h))
                out[p] = (K0 * base.get(p, 0.0) + obs) / (K0 + w.sum())
        return out

    def team_net(sh, mean):
        tot = sum(sh.values())
        val = sum(s_ * (mean[ix[p]] + mean[P + ix[p]]) for p, s_ in sh.items() if p in ix)
        return val + max(0.0, 5.0 - tot) * repl

    game_rows, ply_rows, daily = [], [], []
    last_date = None
    for date, day in reg_games.groupby("DATE", sort=True):
        if last_date is not None:                       # random-walk drift between dates
            gap = (pd.Timestamp(date) - pd.Timestamp(last_date)).days
            S[np.diag_indices(D)] += qd * gap
        last_date = date
        sd_vec = np.sqrt(np.diag(S))
        for t in hist:
            daily.append({"date": date, "team": t, "net": round(team_net(shares(t), m), 3)})
        # ---- pre-game predictions (state BEFORE this date's games) ----
        for r in day.itertuples():
            gid = int(r.GAME_ID)
            row = {"GAME_ID": gid, "date": date, "home": r.HOME, "away": r.AWAY}
            for side, team in (("home", r.HOME), ("away", r.AWAY)):
                sh = shares(team)
                row[f"net_{side}_strict"] = team_net(sh, m)
                shc = shares(team, conditional=True)
                # pre-game availability: P(plays) from games just missed + role (fit on
                # prior seasons only) -> expected presence with fitted redistribution
                h = hist.get(team, [])
                pids_c = list(shc)
                streak = []
                for p in pids_c:
                    if not any(p in g_ for g_ in h):
                        streak.append(-1); continue
                    k_ = 0
                    for g_ in reversed(h):
                        if g_.get(p, 0.0) > 0:
                            break
                        k_ += 1
                    streak.append(k_)
                if avail_model is not None and pids_c:
                    pp = avail_model.predict_proba(avail_features(streak, [shc[p] for p in pids_c]))[:, 1]
                else:
                    pp = np.where(np.array(streak) == 0, 0.9, np.where(np.array(streak) < 0, 0.6, 0.3))
                pav = dict(zip(pids_c, pp))
                if injury_log is not None:      # live: official report beats the streak proxy
                    pav.update({k_: v_ for k_, v_ in ia.availability_override(
                        injury_log, team, f"{date}T22:00:00Z", name_to_pid).items() if k_ in pav})
                row[f"net_{side}_pregame"] = team_net(redistribute(shc, pav), m)
                act = gpres.get((gid, team))
                if act:
                    cond = {p: shc.get(p, 0.0) for p in set(shc) | set(act)}
                    for p in act:
                        if cond[p] <= 0:
                            cond[p] = 0.1          # unseen player: small role
                    row[f"net_{side}_actives"] = team_net(
                        redistribute(cond, {p: 1.0 for p in act}), m)
                else:
                    row[f"net_{side}_actives"] = np.nan
                for p, x in sh.items():
                    if p in ix:
                        ply_rows.append((s, gid, date, p, team, m[ix[p]], m[P + ix[p]],
                                         sd_vec[ix[p]], sd_vec[P + ix[p]], x,
                                         int(bool(act) and p in act)))
            game_rows.append(row)
        # ---- Kalman update with this date's stints ----
        g = stints_by_date.get(date)
        if g is not None and len(g):
            n = len(g)
            H = np.zeros((2 * n, D)); y = np.zeros(2 * n); R = np.zeros(2 * n)
            for k_, (hl, al, poss, yo, yd) in enumerate(zip(g.home, g.away, g.POSS,
                                                            g.Y_OFF_HOME, g.Y_DEF_HOME)):
                for rr, off, de, yy, home in ((2 * k_, hl, al, yo, 1), (2 * k_ + 1, al, hl, yd, 0)):
                    for p in off:
                        H[rr, ix[p]] += 1.0
                    for p in de:
                        H[rr, P + ix[p]] -= 1.0
                    H[rr, 2 * P] = 1.0
                    H[rr, 2 * P + 1] = home
                    y[rr] = yy - L
                    R[rr] = C / max(poss, 0.05)
            cols = np.where(np.abs(H).sum(0) > 0)[0]     # only touched states
            Hc = H[:, cols]
            SHt = S[:, cols] @ Hc.T                       # D x n
            Sm = Hc @ SHt[cols, :] + np.diag(R)
            Kt = np.linalg.solve(Sm, SHt.T)               # n x D  (= K^T)
            m = m + Kt.T @ (y - Hc @ m[cols])
            S = S - SHt @ Kt
            S = 0.5 * (S + S.T)
        # ---- share bookkeeping after the games ----
        for r in day.itertuples():
            gid = int(r.GAME_ID)
            for team in (r.HOME, r.AWAY):
                act = gpres.get((gid, team))
                if act is None:
                    continue
                for p in act:
                    member[p] = team
                roster = {p for p in shares(team)} | set(act)
                hist.setdefault(team, []).append({p: act.get(p, 0.0) for p in roster})
    if verbose:
        print(f"  {s}: {P} players, L={L:.1f}, C={C:.0f}, mu={m[2*P]:+.2f}, h={m[2*P+1]:+.2f}")
    G = pd.DataFrame(game_rows)
    Pf = pd.DataFrame(ply_rows, columns=["season", "GAME_ID", "date", "PLAYER_ID", "team",
                                         "o", "d", "sd_o", "sd_d", "share", "played"])
    return G, Pf, pd.DataFrame(daily)


def availability_panel():
    """Player-game availability panel 2016+ (cached): streak, share-when-playing, played."""
    p = CACHE / "availability_panel.parquet"
    if p.exists():
        P = pd.read_parquet(p)
        if P.season.min() <= 2016:
            return P
    import sys as _s
    _s.path.insert(0, str(RAPM))
    import availability_research as ar
    ctx = ms._Ctx()
    P = pd.concat([ar.season_games(ctx, s) for s in sorted(ctx.tgs) if s >= 2016], ignore_index=True)
    P.to_parquet(p)
    return P


def main(seasons=range(2019, 2028)):
    data = pi.BookerData()
    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    allp = ms.load()
    panel = availability_panel()
    for s in seasons:
        if s not in data.GAMES or s - 1 not in data.STINTS or not (allp.season == s).any():
            print(f"  {s}: skipped"); continue
        proj = allp[allp.season == s]
        repl = float(proj.repl.iloc[0])
        log = None
        if s not in data.STINTS:        # live/forward season: use collected injury reports
            try:
                from . import injury_availability as ia
                log = ia.load_log()
            except Exception as exc:     # no log reachable -> streak proxy only
                print(f"  {s}: injury log unavailable ({exc})")
        G, Pf, Dd = run_season(data, s, rat, proj, repl, avail_model=fit_availability(panel, s),
                               injury_log=log)
        G.to_csv(CACHE / f"seq_games_{s}.csv", index=False)
        Pf.to_parquet(CACHE / f"seq_players_{s}.parquet")
        Dd.to_csv(CACHE / f"seq_team_daily_{s}.csv", index=False)


if __name__ == "__main__":
    ss = [int(a) for a in sys.argv[1:]]
    main(ss or range(2019, 2028))
