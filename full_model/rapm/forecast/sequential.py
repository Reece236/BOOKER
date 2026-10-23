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
# PRIOR SEED (2026-10-22 re-tune, see HANDOFF 0.8): prior mean = aged row s-1 +
# PROJ_LAMBDA * (BOOKER-PROJ for s - aged total), the delta split O/D in proportion to the
# prior variances. PRIOR_SD_MULT scales the row's sd (overlay) before ETA is added;
# _SD_OVERRIDE {(season, pid): (sd_o, sd_d)} lets a harness swap the sd source.
PROJ_LAMBDA = 0.5     # gate 2026-10-22 (HANDOFF 0.8): mid rho vs pure +.013 SEL(2020-23) / +.019 TEST(2024-26)
PRIOR_SD_MULT = 1.0
PROJ_MATCH_SPREAD = False   # rescale PROJ's (shrunk) spread to the aged ratings' before blending
_SD_OVERRIDE = {}
_PROJ_CACHE = {}
# Minutes redistribution when players sit (availability_research.py, fit on realized
# per-game presence 2019-21, tested 2022-26): absent minutes go to available teammates
# with weight c^ALPHA (c = share when playing). ALPHA < 1 -> the deeper bench absorbs
# relatively more; a same-position term fit to zero. Test presence RMSE .1352 -> .1315.
ALPHA = 0.4
# BOX-SCORE MEASUREMENT CHANNEL (audit #2: stint-only evidence learns too slowly -- a full MVP
# season moved Curry +3.8). Each player-game adds  f(box per 100) = o + d + b_p + e,
# e ~ N(0, BOX_KAPPA * sigma2 / poss),  b_p ~ N(0, BOX_TAU^2) a persistent per-player box
# misfit state (defensive specialists etc. -- without it the filter would converge to the
# box score). f, sigma2 fit on seasons < s only (forecast.box_games.fit_box_map).
# RESULT (box_tune.py): OFF. Final spec = within-team contrasts + misfit prior from last
# season. Tuned 2019-21 (tau 3, kappa 1): LL .6199->.6194, mid rho .406->.410. TEST 2022-26:
# pregame LL .6134->.6131 (nil), mid rho .423->.418, late .385->.367 (worse). Earlier specs
# (raw rows, zero-mean misfit) were clearly harmful (double-counted team shooting luck).
BOX_ON = False
BOX_TAU = 3.0
BOX_KAPPA = 1.0
BOX_MIN_POSS = 5.0
# Leverage-scaled stint precision (2026-10-22b gate, HANDOFF 0.9): R = C / (POSS * c),
# c = forecast.leverage.stint_weight (clip(1 + LEV_KAPPA*(LI-1), .5, 1.3), poss-mean 1). 0 = off.
LEV_KAPPA = 0.0
UNRATED_PRIOR = "repl"  # "repl" (old: repl/2 per side) | "wf" (minutes_share.unrated_default)
AGE_CARRY_FIX = True   # age short-season fallback rows from carried_from (False = old 1-year aging)
ROT_MIN = 0.12     # actives mode: min share-when-playing (~6 min/g) to count as an active


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


def _proj_for(s):
    """{pid: BOOKER-PROJ total for season s} (walk-forward: built from data through s-1)."""
    if s not in _PROJ_CACHE:
        p = CACHE / "booker_proj.csv"
        if p.exists():
            d = pd.read_csv(p)
            d = d[d.proj_for_season == s]
            _PROJ_CACHE[s] = dict(zip(d.PLAYER_ID.astype(int), d.proj_impact.astype(float)))
        else:
            _PROJ_CACHE[s] = {}
    return _PROJ_CACHE[s]


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


def _cutoff(r, date):
    """Injury-report cutoff for a game: 1h before the real tip (matinees tip 1-3:30pm ET, so a
    fixed 6pm ET read confirmed inactives); 22:00Z when no real tip time is known."""
    tip = getattr(r, "TIP_UTC", None)
    if isinstance(tip, str) and not tip.endswith("T00:00Z") and getattr(r, "PLACEHOLDER", 0) != 1:
        try:
            return (pd.Timestamp(tip) - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass
    return f"{date}T22:00:00Z"


def run_season(data, s, rat, proj, repl, verbose=True, avail_model=None, injury_log=None,
               box=None, box_tau=None, box_kappa=None):
    """`box` = (games_box_df for season s, fitted map, sigma2) or None (stint-only)."""
    box_tau = BOX_TAU if box_tau is None else box_tau
    box_kappa = BOX_KAPPA if box_kappa is None else box_kappa
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
    r1 = pi.ratings_through(s - 1, rat).set_index("PLAYER_ID")     # + lost-season carries
    a = unc.attach_attrs(r1.reset_index()[["player", "season"]].assign(minutes=0))
    ages = dict(zip(r1.index, a.age.values))
    pids = set(proj.PLAYER_ID.astype(int))
    if st is not None:
        for hl, al in zip(st.home, st.away):
            pids.update(hl); pids.update(al)
    pids = sorted(pids)
    P = len(pids)
    ix = {p: i for i, p in enumerate(pids)}
    NB = P if box is not None else 0                  # per-player box-misfit states
    D = 2 * P + 2 + NB
    m = np.zeros(D)
    v = np.zeros(D)
    pj = _proj_for(s) if PROJ_LAMBDA else {}
    if pj and PROJ_MATCH_SPREAD:
        both = [p for p in pj if p in r1.index]
        if len(both) > 50:
            a_ = np.array([float(r1.at[p, "impact_off"] + r1.at[p, "impact_def"]) for p in both])
            b_ = np.array([pj[p] for p in both])
            k_ = a_.std() / max(b_.std(), 1e-6)
            mb = b_.mean()
            pj = {p: mb + k_ * (v - mb) for p, v in pj.items()}
    for p, i in ix.items():
        if p in r1.index:
            o, dd = float(r1.at[p, "impact_off"]), float(r1.at[p, "impact_def"])
            ag = ages.get(p)
            if ag == ag and ag is not None:            # aging to season s, split O/D
                # short-season fallback rows (ratings_through) were carried from s-2/s-3:
                # age them from the season they were fit through, not from s-1 (2026-10-22b)
                cf = r1.at[p, "carried_from"] if "carried_from" in r1.columns else np.nan
                yrs = (s - int(cf)) if (cf == cf and AGE_CARRY_FIX) else 1
                da = pi.AGE_QUAD * ((ag + 1 - pi.AGE_PEAK) ** 2 - (ag + 1 - yrs - pi.AGE_PEAK) ** 2)
                o += da / 2; dd += da / 2
            so_, sd_ = _SD_OVERRIDE.get((s - 1, p), (float(r1.at[p, "sd_off"]), float(r1.at[p, "sd_def"])))
            v[i] = (PRIOR_SD_MULT * so_) ** 2 + ETA ** 2
            v[P + i] = (PRIOR_SD_MULT * sd_) ** 2 + ETA ** 2
            if p in pj:                                # shrink toward BOOKER-PROJ (fallback: aged)
                delta = PROJ_LAMBDA * (pj[p] - (o + dd))
                wo = v[i] / (v[i] + v[P + i])
                o += delta * wo; dd += delta * (1.0 - wo)
            m[i], m[P + i] = o, dd
        else:
            # unrated prior: repl/2 per side (old); UNRATED_PRIOR="wf" -> the walk-forward
            # unrated default (minutes_share.unrated_default) split evenly
            m[i] = m[P + i] = (ms.unrated_default(s, rat) if UNRATED_PRIOR == "wf" else repl) / 2.0
            v[i] = v[P + i] = NEW_SD ** 2
    m[2 * P], v[2 * P] = 0.0, MU_SD ** 2
    m[2 * P + 1], v[2 * P + 1] = H_MEAN, H_SD ** 2
    if NB:
        v[2 * P + 2:] = box_tau ** 2
    S = np.diag(v)
    qd = np.full(D, Q_DAY); qd[2 * P] = Q_MU; qd[2 * P + 1] = 0.0
    if NB:
        qd[2 * P + 2:] = 0.0                            # misfit persists within the season
        bx, bmap, bsig2, bprev = box
        bx = bx[bx.poss >= BOX_MIN_POSS]
        from . import box_games as bg
        # misfit prior: the box's persistent over/under-statement of a player is KNOWN from
        # last season (f(last season's box) - prior rating); start b_p there so the box only
        # carries CHANGES in production. Players without a real s-1 box sample start at 0.
        if bprev is not None and len(bprev):
            fprev = dict(zip(bprev.PLAYER_ID.astype(int), bmap.predict(bg.rates(bprev))))
            for p_, i_ in ix.items():
                if p_ in fprev and p_ in r1.index:
                    m[2 * P + 2 + i_] = fprev[p_] - (m[i_] + m[P + i_])   # so E[y] = f(box_{s-1})
        bx = bx.assign(yb=bmap.predict(bg.rates(bx)))
        box_by_gid = {int(g_): list(zip(d_.PLAYER_ID.astype(int), d_.yb, d_.poss))
                      for g_, d_ in bx.groupby("GAME_ID")}
    else:
        box_by_gid = {}

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
    if st is not None:
        if LEV_KAPPA:
            from . import leverage as lev
            st = st.assign(LEVC=lev.stint_weight(s, LEV_KAPPA, st.POSS.to_numpy(), st.GAME_ID.to_numpy()))
        else:
            st = st.assign(LEVC=1.0)
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
                        injury_log, team, _cutoff(r, date), name_to_pid).items() if k_ in pav})
                row[f"net_{side}_pregame"] = team_net(redistribute(shc, pav), m)
                act = gpres.get((gid, team))
                if act:
                    # ACTIVES = rotation players who suited up. Only players with a real
                    # role (share-when-playing >= ROT_MIN) count, so who got garbage-time /
                    # coach's-decision minutes -- in-game information correlated with the
                    # margin (audit: actives count r=.57 with |margin|) -- carries no signal.
                    # A player traded in keeps his share-when-playing from his old team.
                    cond = dict(shc)
                    for p in act:
                        if cond.get(p, 0.0) <= 0:
                            old_t = member.get(p)
                            cond[p] = (shares(old_t, conditional=True).get(p, 0.0)
                                       if old_t and old_t != team else 0.0)
                    on = {p: float(p in act and cond.get(p, 0.0) >= ROT_MIN) for p in cond}
                    row[f"net_{side}_actives"] = team_net(redistribute(
                        {p: c_ for p, c_ in cond.items() if c_ > 0}, on), m)
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
        # WITHIN-TEAM box contrasts: centre each player's f(box) on his teammates'
        # poss-weighted mean that game. Team shooting luck is already in the stint margins;
        # raw box rows counted it twice and were correlated across all ten players.
        brows = []                                       # (pids, weights, y, poss)
        for r_ in day.itertuples():
            bg_ = {p_: (yb_, ps_) for (p_, yb_, ps_) in box_by_gid.get(int(r_.GAME_ID), []) if p_ in ix}
            if not bg_:
                continue
            for team_ in (r_.HOME, r_.AWAY):
                act_ = gpres.get((int(r_.GAME_ID), team_)) or {}
                mem = [p_ for p_ in act_ if p_ in bg_]
                if len(mem) < 4:
                    continue
                ps = np.array([bg_[p_][1] for p_ in mem]); w_ = ps / ps.sum()
                yv = np.array([bg_[p_][0] for p_ in mem]); ybar = float(w_ @ yv)
                for j_, p_ in enumerate(mem):
                    brows.append((mem, w_, j_, yv[j_] - ybar, ps[j_]))
        if g is not None and len(g):
            n = len(g)
            nb = len(brows)
            H = np.zeros((2 * n + nb, D)); y = np.zeros(2 * n + nb); R = np.zeros(2 * n + nb)
            for j_, (mem, w_, jj, yc, ps_) in enumerate(brows):
                rr = 2 * n + j_
                for k_, q_ in enumerate(mem):
                    c_ = (1.0 if k_ == jj else 0.0) - w_[k_]
                    iq = ix[q_]
                    H[rr, iq] += c_; H[rr, P + iq] += c_; H[rr, 2 * P + 2 + iq] += c_
                y[rr] = yc
                R[rr] = box_kappa * bsig2 / ps_
            for k_, (hl, al, poss, yo, yd, lc_) in enumerate(zip(g.home, g.away, g.POSS,
                                                                 g.Y_OFF_HOME, g.Y_DEF_HOME, g.LEVC)):
                for rr, off, de, yy, home in ((2 * k_, hl, al, yo, 1), (2 * k_ + 1, al, hl, yd, 0)):
                    for p in off:
                        H[rr, ix[p]] += 1.0
                    for p in de:
                        H[rr, P + ix[p]] -= 1.0
                    H[rr, 2 * P] = 1.0
                    H[rr, 2 * P + 1] = home
                    y[rr] = yy - L
                    R[rr] = C / (max(poss, 0.05) * lc_)
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


_BOX = {}


def box_inputs(s, rat):
    """(season-s box lines, map fit on seasons < s, sigma2) or None."""
    if not BOX_ON:
        return None
    from . import box_games as bg
    if "B" not in _BOX:
        p = CACHE / "box_games.parquet"
        _BOX["B"] = pd.read_parquet(p) if p.exists() else None
    B = _BOX["B"]
    if B is None:
        return None
    prior = [x for x in sorted(B.season.unique()) if x < s]
    if len(prior) < 3:
        return None
    bmap, _, sig2 = bg.fit_box_map(B, rat, prior)
    cur = B[B.season == s]
    pv = B[B.season == s - 1].groupby("PLAYER_ID")[bg.FEATS + ["poss"]].sum().reset_index()
    pv = pv[pv.poss >= bg.MIN_POSS_FIT / 2]
    return (cur, bmap, sig2, pv)


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
        if s not in data.STINTS or s in data.PARTIAL:   # live / in-progress: collected injury reports
            try:
                from . import injury_availability as ia
                log = ia.load_log()
            except Exception as exc:     # no log reachable -> streak proxy only
                print(f"  {s}: injury log unavailable ({exc})")
        G, Pf, Dd = run_season(data, s, rat, proj, repl, avail_model=fit_availability(panel, s),
                               injury_log=log, box=box_inputs(s, rat))
        G.to_csv(CACHE / f"seq_games_{s}.csv", index=False)
        Pf.to_parquet(CACHE / f"seq_players_{s}.parquet")
        Dd.to_csv(CACHE / f"seq_team_daily_{s}.csv", index=False)


if __name__ == "__main__":
    ss = [int(a) for a in sys.argv[1:]]
    main(ss or range(2019, 2028))
