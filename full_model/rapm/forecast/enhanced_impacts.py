"""
Bayesian player impacts with offensive/defensive split and teammate-fit credit sharing.

Stints only carry combined net margin, so we:

  1. Fit standard additive ridge RAPM (total impact).
  2. Split total -> offensive / defensive via Bayesian shrinkage toward box-score
     offensiveWS / defensiveWS shares (with minutes-based credibility).
  3. Estimate sparse same-team pair synergies from stint residuals (ridge on top
     co-minute pairs only).
  4. Redistribute offensive credit from high-finisher to high-creator teammates when
     their pair shows positive offensive synergy (e.g. roll-man benefiting from PG
     creation gets less offensive credit; the PG gets more).

Public API used by trade_sim, generate_charts, and preseason.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

from . import player_impacts as pi

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
PLAYER_DATA = pi.PLAYER_DATA

# pair synergy ridge; higher = more shrinkage toward zero interaction
PAIR_ALPHA = 8000.0
MIN_PAIR_POSS = 400.0          # minimum co-possessions to estimate a pair
MAX_PAIRS = 800                # cap sparse pair features per season fit
CREDIT_SHARE = 0.45            # fraction of positive offensive synergy shifted to creator
MAX_OFF_SHIFT = 0.30           # max fraction of finisher offensive impact redistributed
OFF_DEF_PRIOR_K = 180.0
# Bayesian predictive WAA@32: full 82-game season at 32 mpg with extra shrinkage
# toward box prior for low-minute players (separate from in-model ridge prior).
STANDARD_MPG = 32.0
STANDARD_GP = 82
STANDARD_SEASON_MIN = STANDARD_MPG * STANDARD_GP
PREDICTIVE_K = 1000.0
WORKLOAD_FLOOR = 1800.0  # minutes before trusting full 32-mpg extrapolation
RANK_SHRINK_K = 700.0    # minutes credibility for main WAA rank (actual minutes)


def rank_cred(minutes, k=RANK_SHRINK_K):
    """Sample-size credibility for main WAA at observed minutes."""
    if minutes <= 0:
        return 0.0
    return minutes / (minutes + k)


def rank_impact(fitted, prior, minutes, k=RANK_SHRINK_K):
    """Shrink fitted impact toward prior for leaderboard WAA."""
    cred = rank_cred(minutes, k=k)
    return cred * fitted + (1.0 - cred) * prior


def predictive_cred(minutes, k=PREDICTIVE_K, workload_floor=WORKLOAD_FLOOR):
    """Combined sample-size and workload credibility for 32-mpg projection."""
    if minutes <= 0:
        return 0.0
    cred_sample = minutes / (minutes + k)
    cred_workload = min(1.0, minutes / workload_floor)
    return cred_sample * cred_workload


def predictive_impact(fitted, prior, minutes, k=PREDICTIVE_K):
    """Shrink fitted impact toward prior; small samples regress harder."""
    cred = predictive_cred(minutes, k=k)
    return cred * fitted + (1.0 - cred) * prior


# When roster-aggregated net overshoots observed team net, shrink positive
# excess above box prior (bad-team carry job / weak-teammate collinearity).
RECONCILE_MIN_GAP = 1.0
RECONCILE_BUFFER = 0.5
RECONCILE_MAX_EXCESS = 0.70
RECONCILE_PASSES = 3
BAD_TEAM_NET = -3.0
OVER_SHOOT_MULT = 1.25   # extra shrink when roster sum >> actual (e.g. HOU)


def reconcile_team_nets(data, season, total, off, def_, prior, last_age, minutes=None):
    """Shrink inflated impacts when roster sum overshoots observed team net."""
    pl = data.PLAYERS.get(season)
    tm = data.TEAMS.get(season)
    if pl is None or tm is None:
        return total, off, def_

    mins = minutes if minutes is not None else dict(zip(pl.PLAYER_ID, pl.MINUTES))
    act_net = {int(t): float(v) for t, v in zip(tm.TEAM_ID, tm.ACTUAL_NET)
               if pd.notna(v)}

    total, off, def_ = dict(total), dict(off), dict(def_)

    for tid, actual in act_net.items():
        roster = [(int(pid), float(mins.get(pid, 0)))
                  for pid, t in zip(pl.PLAYER_ID, pl.TEAM_ID) if int(t) == tid]
        roster = [(p, m) for p, m in roster if m > 0]
        if not roster:
            continue
        tmin = sum(m for _, m in roster)

        def roster_net(imp):
            s = 0.0
            for pid, mn in roster:
                v = pi.aged_value(imp, pid, last_age, season)
                s += v * (mn / (tmin / 5.0))
            return s

        pred = roster_net(total)
        gap = pred - (actual + RECONCILE_BUFFER)
        if gap < RECONCILE_MIN_GAP:
            continue

        bad_mult = 1.0
        if actual < BAD_TEAM_NET:
            bad_mult = 1.0 + 0.35 * min((BAD_TEAM_NET - actual) / 10.0, 1.0)
        if gap > 4.0:
            bad_mult *= OVER_SHOOT_MULT

        # distribute gap removal across positive excess, weighted by minutes
        weights = []
        for pid, mn in roster:
            v = pi.aged_value(total, pid, last_age, season)
            pr = prior.get(pid, pi.PRIOR_BASE)
            excess = max(0.0, v - pr)
            if excess > 0:
                weights.append((pid, excess * mn))

        wsum = sum(w for _, w in weights)
        if wsum <= 0:
            continue

        remove = gap * bad_mult
        for pid, w in weights:
            v = total.get(pid, pi.PRIOR_BASE)
            pr = prior.get(pid, pi.PRIOR_BASE)
            excess = max(0.0, v - pr)
            if excess <= 0:
                continue
            share = remove * (w / wsum)
            cut = min(share, excess * RECONCILE_MAX_EXCESS)
            o, d = off.get(pid, 0.0), def_.get(pid, 0.0)
            if abs(v) > 1e-6:
                off[pid] = o - cut * (o / v)
                def_[pid] = d - cut * (d / v)
            total[pid] = v - cut

    return total, off, def_


def norm_name(s):
    return pi.norm_name(s)


@dataclass
class EnhancedImpacts:
    total: dict
    off: dict
    def_: dict
    prior: dict
    last_age: dict
    synergies: dict          # (pid_lo, pid_hi) -> pts/100 poss synergy
    roles: dict              # pid -> {creator, finisher, defender}


def _load_box_table():
    bk = pd.read_csv(PLAYER_DATA).dropna(subset=["box"])
    bk["nm"] = bk.playerName.map(norm_name)
    return bk


def box_off_def_shares(data, target_season, pids):
    """Bayesian offensive share of total impact per player id."""
    bk = _load_box_table()
    # latest season at or before target for each name
    sub = bk[bk.season <= target_season]
    latest = (sub.sort_values("season")
              .groupby("nm", as_index=False)
              .tail(1)
              .set_index("nm"))
    pid_nm = {}
    for s in data.seasons:
        if s > target_season:
            continue
        pl = data.PLAYERS.get(s)
        if pl is None:
            continue
        for pid, nm in zip(pl.PLAYER_ID, pl.NAME):
            pid_nm[int(pid)] = norm_name(nm)

    shares = {}
    for pid in pids:
        nm = pid_nm.get(pid)
        if nm is None or nm not in latest.index:
            shares[pid] = 0.5
            continue
        row = latest.loc[nm]
        ows = max(float(row.get("offensiveWS", 0) or 0), 0.05)
        dws = max(float(row.get("defensiveWS", 0) or 0), 0.05)
        ast = float(row.get("total_assists", 0) or 0)
        mp = max(float(row.get("minutesPlayed", 1) or 1), 1.0)
        blk = float(row.get("total_blocks", 0) or 0)
        stl = float(row.get("total_steals", 0) or 0)
        # role-tilted split: creators tilt offensive; stoppers tilt defensive
        ast_rate = ast / mp
        stop_rate = (blk + stl) / mp
        raw_off = ows / (ows + dws)
        raw_off += 0.08 * min(ast_rate * 80, 1.0)
        raw_off -= 0.06 * min(stop_rate * 120, 1.0)
        shares[pid] = float(np.clip(raw_off, 0.22, 0.78))
    return shares


def role_indices(data, target_season, pids):
    """Creator / finisher / defender indices in [0, 1] from box rates."""
    bk = _load_box_table()
    sub = bk[bk.season <= target_season]
    latest = (sub.sort_values("season")
              .groupby("nm", as_index=False)
              .tail(1)
              .set_index("nm"))
    pid_nm = {}
    for s in data.seasons:
        if s > target_season:
            continue
        pl = data.PLAYERS.get(s)
        if pl is None:
            continue
        for pid, nm in zip(pl.PLAYER_ID, pl.NAME):
            pid_nm[int(pid)] = norm_name(nm)

    roles = {}
    ast_rates, usg_rates, stop_rates = [], [], []
    raw = {}
    for pid in pids:
        nm = pid_nm.get(pid)
        if nm is None or nm not in latest.index:
            continue
        row = latest.loc[nm]
        mp = max(float(row.get("minutesPlayed", 1) or 1), 1.0)
        ast_r = float(row.get("total_assists", 0) or 0) / mp
        usg = float(row.get("usagePercent", 0.2) or 0.2)
        pts = float(row.get("total_points", 0) or 0) / mp
        blk = float(row.get("total_blocks", 0) or 0) / mp
        stl = float(row.get("total_steals", 0) or 0) / mp
        ast_rates.append(ast_r)
        usg_rates.append(usg)
        stop_rates.append(blk + stl)
        raw[pid] = (ast_r, usg, pts, blk + stl)

    def rank_scale(vals, v, invert=False):
        if not vals:
            return 0.5
        arr = np.array(vals)
        p = (arr < v).mean() if not invert else (arr > v).mean()
        return float(np.clip(p, 0.05, 0.95))

    for pid in pids:
        if pid not in raw:
            roles[pid] = {"creator": 0.33, "finisher": 0.33, "defender": 0.33}
            continue
        ast_r, usg, pts, stop = raw[pid]
        pts_list = [v[2] for v in raw.values() if len(v) >= 4]
        creator = 0.55 * rank_scale(ast_rates, ast_r) + 0.45 * rank_scale(ast_rates, ast_r)
        finisher = 0.5 * rank_scale(usg_rates, usg) + 0.5 * (1 - rank_scale(ast_rates, ast_r, invert=True))
        finisher *= 0.6 + 0.4 * rank_scale(pts_list, pts)
        defender = rank_scale(stop_rates, stop)
        roles[pid] = {
            "creator": float(np.clip(creator, 0, 1)),
            "finisher": float(np.clip(finisher, 0, 1)),
            "defender": float(np.clip(defender, 0, 1)),
        }
    return roles


def split_off_def(total, shares, minutes):
    """Split total impact into offensive and defensive components."""
    off, def_ = {}, {}
    for pid, tot in total.items():
        sh = shares.get(pid, 0.5)
        mn = minutes.get(pid, 0.0)
        cred = mn / (mn + OFF_DEF_PRIOR_K) if mn > 0 else 0.0
        eff = cred * sh + (1 - cred) * 0.5
        off[pid] = tot * eff
        def_[pid] = tot * (1 - eff)
    return off, def_


def _pair_key(a, b):
    return (a, b) if a < b else (b, a)


def estimate_pair_synergies(data, train_seasons, target_season, impact, extra_stints=None):
    """Sparse ridge on stint residuals for frequent same-team pairs."""
    co_poss = {}
    stint_rows = []

    def ingest(hl, al, poss, y):
        pred = 0.0
        for p in hl:
            pred += impact.get(p, pi.PRIOR_BASE)
        for p in al:
            pred -= impact.get(p, pi.PRIOR_BASE)
        resid = y - pred
        home_set = set(hl)
        away_set = set(al)
        for side, sign in ((home_set, 1), (away_set, -1)):
            lst = sorted(side)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    k = _pair_key(lst[i], lst[j])
                    co_poss[k] = co_poss.get(k, 0.0) + poss
            stint_rows.append((resid, poss, lst, sign))

    for s in train_seasons:
        d = data.STINTS[s]
        for hl, al, poss, y in zip(d.home, d.away, d.POSS, d.Y):
            ingest(hl, al, poss, y)
    if extra_stints is not None and len(extra_stints):
        for hl, al, poss, y in zip(extra_stints.home, extra_stints.away,
                                   extra_stints.POSS, extra_stints.Y):
            ingest(hl, al, poss, y)

    pairs = sorted([k for k, v in co_poss.items() if v >= MIN_PAIR_POSS],
                   key=lambda k: -co_poss[k])[:MAX_PAIRS]
    if not pairs:
        return {}

    pcol = {k: i for i, k in enumerate(pairs)}
    rows, cols, vals, ys, ws = [], [], [], [], []
    ri = 0
    for resid, poss, lst, sign in stint_rows:
        active = []
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                k = _pair_key(lst[i], lst[j])
                if k in pcol:
                    active.append(k)
        if not active:
            continue
        for k in active:
            rows.append(ri)
            cols.append(pcol[k])
            vals.append(sign)
        ys.append(resid)
        ws.append(poss)
        ri += 1
    if ri < 20:
        return {}

    X = csr_matrix((vals, (rows, cols)), shape=(ri, len(pairs)))
    y = np.array(ys)
    w = np.array(ws)
    ridge = Ridge(alpha=PAIR_ALPHA, fit_intercept=False)
    ridge.fit(X, y, sample_weight=w)
    return {pairs[i]: float(ridge.coef_[i]) for i in range(len(pairs))}


def redistribute_off_credit(off, roles, synergies, co_poss):
    """Shift offensive credit from finishers to creators on positive pair synergy."""
    off = dict(off)
    for (a, b), syn in synergies.items():
        if syn <= 0:
            continue
        ra, rb = roles.get(a, {}), roles.get(b, {})
        ca, fa = ra.get("creator", 0), ra.get("finisher", 0)
        cb, fb = rb.get("creator", 0), rb.get("finisher", 0)
        poss = co_poss.get((a, b), co_poss.get((b, a), 0))
        if poss < MIN_PAIR_POSS:
            continue
        # identify creator-finisher orientation
        if ca >= cb and fb >= fa:
            creator, finisher = a, b
        elif cb >= ca and fa >= fb:
            creator, finisher = b, a
        else:
            continue
        fin_off = off.get(finisher, 0.0)
        if fin_off <= 0:
            continue
        shift = min(syn * CREDIT_SHARE, fin_off * MAX_OFF_SHIFT, 2.5)
        if shift <= 0:
            continue
        off[finisher] = fin_off - shift
        off[creator] = off.get(creator, 0.0) + shift
    return off


def _co_possessions(data, train_seasons, extra_stints=None):
    co = {}
    def bump(hl, al, poss):
        for side in (set(hl), set(al)):
            lst = sorted(side)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    k = _pair_key(lst[i], lst[j])
                    co[k] = co.get(k, 0.0) + poss
    for s in train_seasons:
        d = data.STINTS[s]
        for hl, al, poss in zip(d.home, d.away, d.POSS):
            bump(hl, al, poss)
    if extra_stints is not None and len(extra_stints):
        for hl, al, poss in zip(extra_stints.home, extra_stints.away, extra_stints.POSS):
            bump(hl, al, poss)
    return co


def _renormalize_off_def(total, off, def_):
    """Keep off/def splits while forcing off + def = total per player."""
    off, def_ = dict(off), dict(def_)
    for pid, t in total.items():
        o, d = off.get(pid, 0.0), def_.get(pid, 0.0)
        s = o + d
        if abs(s) < 1e-6:
            off[pid] = t * 0.5
            def_[pid] = t * 0.5
        else:
            scale = t / s
            off[pid] = o * scale
            def_[pid] = d * scale
    return off, def_


def build_enhanced(data, train_seasons, target_season, alpha=None,
                   extra_stints=None, extra_weight=1.0):
    """Net RAPM (validated) -> team reconcile -> O/D split + credit sharing."""
    alpha = alpha or pi.pick_alpha(data, train_seasons)
    total, prior, last_age = pi.build_impacts(
        data, train_seasons, target_season, alpha,
        extra_stints=extra_stints, extra_weight=extra_weight)
    pids = list(total.keys())

    minutes = {}
    for s in train_seasons + ([target_season] if target_season in data.PLAYERS else []):
        pl = data.PLAYERS.get(s)
        if pl is None:
            continue
        for pid, mn in zip(pl.PLAYER_ID, pl.MINUTES):
            minutes[int(pid)] = minutes.get(int(pid), 0.0) + float(mn)

    shares = box_off_def_shares(data, target_season, pids)
    off, def_ = split_off_def(total, shares, minutes)

    if target_season in data.PLAYERS:
        for _ in range(RECONCILE_PASSES):
            total, off, def_ = reconcile_team_nets(
                data, target_season, total, off, def_,
                prior, last_age, minutes)

    roles = role_indices(data, target_season, pids)
    co = _co_possessions(data, train_seasons, extra_stints)
    synergies = estimate_pair_synergies(
        data, train_seasons, target_season, total, extra_stints=extra_stints)
    off = redistribute_off_credit(off, roles, synergies, co)
    off, def_ = _renormalize_off_def(total, off, def_)

    return EnhancedImpacts(
        total=dict(total),
        off=off,
        def_=def_,
        prior=prior,
        last_age=last_age,
        synergies=synergies,
        roles=roles,
    )


def aggregate_off_def(data, enh, season, target_season=None, minutes=None):
    """Team-level offensive and defensive net ratings."""
    pl = data.PLAYERS[season]
    mins = minutes if minutes is not None else dict(zip(pl.PLAYER_ID, pl.MINUTES))
    tmin = {}
    for pid, tid in zip(pl.PLAYER_ID, pl.TEAM_ID):
        tmin[tid] = tmin.get(tid, 0.0) + mins.get(pid, 0.0)

    off_net, def_net, tot_net = {}, {}, {}
    ts = target_season or season
    for pid, tid in zip(pl.PLAYER_ID, pl.TEAM_ID):
        if tid not in tmin or tmin[tid] <= 0:
            continue
        pres = mins.get(pid, 0.0) / (tmin[tid] / 5.0)
        o = pi.aged_value(enh.off, pid, enh.last_age, ts) if ts else enh.off.get(pid, 0)
        d = pi.aged_value(enh.def_, pid, enh.last_age, ts) if ts else enh.def_.get(pid, 0)
        t = pi.aged_value(enh.total, pid, enh.last_age, ts) if ts else enh.total.get(pid, 0)
        off_net[tid] = off_net.get(tid, 0.0) + o * pres
        def_net[tid] = def_net.get(tid, 0.0) + d * pres
        tot_net[tid] = tot_net.get(tid, 0.0) + t * pres
    return off_net, def_net, tot_net


def player_waa_components(data, season, k_wins, enh, minutes=None):
    """Per-player offensive/defensive/total WAA wins on a roster."""
    pl = data.PLAYERS[season]
    mins = minutes if minutes is not None else dict(zip(pl.PLAYER_ID, pl.MINUTES))
    tmin = {}
    for pid, tid in zip(pl.PLAYER_ID, pl.TEAM_ID):
        tmin[tid] = tmin.get(tid, 0.0) + mins.get(pid, 0.0)

    pids = [int(pid) for pid in pl.PLAYER_ID]
    shares = box_off_def_shares(data, season, pids)

    rows = []
    for pid, nm, tid, mn in zip(pl.PLAYER_ID, pl.NAME, pl.TEAM_ID, pl.MINUTES):
        if tid not in tmin or tmin[tid] <= 0 or mn < 1:
            continue
        pid = int(pid)
        mn_obs = float(mins.get(pid, mn))
        pres = mn_obs / (tmin[tid] / 5.0)
        pres_32 = STANDARD_SEASON_MIN / (tmin[tid] / 5.0)
        o = pi.aged_value(enh.off, pid, enh.last_age, season)
        d = pi.aged_value(enh.def_, pid, enh.last_age, season)
        t = pi.aged_value(enh.total, pid, enh.last_age, season)
        prior_t = pi.aged_value(enh.prior, pid, enh.last_age, season)
        sh = shares.get(pid, 0.5)
        prior_o = prior_t * sh
        prior_d = prior_t * (1.0 - sh)
        rank_o = rank_impact(o, prior_o, mn_obs)
        rank_d = rank_impact(d, prior_d, mn_obs)
        rank_t = rank_o + rank_d
        pred_o = predictive_impact(o, prior_o, mn_obs)
        pred_d = predictive_impact(d, prior_d, mn_obs)
        pred_t = pred_o + pred_d
        ab = data.abbr_of.get(tid, "?")
        rows.append({
            "pid": pid, "player": nm, "team": ab,
            "minutes": mn_obs,
            "impact_off": round(o, 2), "impact_def": round(d, 2), "impact_total": round(t, 2),
            "waa_off": round(k_wins * rank_o * pres, 2),
            "waa_def": round(k_wins * rank_d * pres, 2),
            "waa_total": round(k_wins * rank_t * pres, 2),
            "waa32_off": round(k_wins * pred_o * pres_32, 2),
            "waa32_def": round(k_wins * pred_d * pres_32, 2),
            "waa32_total": round(k_wins * pred_t * pres_32, 2),
        })
    return rows
