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
  pregame   team net = sum(share * (o + d)) + fill * repl; two availability modes:
              strict  : shares as estimated -- uses nothing about tonight's game
              actives : shares renormalized over the players who suit up tonight (the
                        pre-tip injury-report proxy; still no minutes from the game)

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


def run_season(data, s, rat, proj, repl, verbose=True):
    reg_games = data.GAMES[s]
    reg_games = reg_games[reg_games.SEASON_TYPE == "Regular Season"].sort_values(["DATE", "GAME_ID"])
    st = data.STINTS.get(s)
    st = st[st.GAME_ID.astype(str).str.startswith("2")] if st is not None else None
    prev = data.STINTS[s - 1]
    L = _league(prev)
    C = _noise_c(prev, L)

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
                act = gpres.get((gid, team))
                if act:
                    shc = shares(team, conditional=True)
                    sa = {p: shc.get(p, 0.0) for p in act}
                    if sum(sa.values()) <= 0:
                        sa = {p: 1.0 for p in act}
                    k = 5.0 / sum(sa.values())
                    row[f"net_{side}_actives"] = team_net({p: x * k for p, x in sa.items()}, m)
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


def main(seasons=range(2019, 2028)):
    data = pi.BookerData()
    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    allp = ms.load()
    for s in seasons:
        if s not in data.GAMES or s - 1 not in data.STINTS or not (allp.season == s).any():
            print(f"  {s}: skipped"); continue
        proj = allp[allp.season == s]
        repl = float(proj.repl.iloc[0])
        G, Pf, Dd = run_season(data, s, rat, proj, repl)
        G.to_csv(CACHE / f"seq_games_{s}.csv", index=False)
        Pf.to_parquet(CACHE / f"seq_players_{s}.parquet")
        Dd.to_csv(CACHE / f"seq_team_daily_{s}.csv", index=False)


if __name__ == "__main__":
    ss = [int(a) for a in sys.argv[1:]]
    main(ss or range(2019, 2028))
