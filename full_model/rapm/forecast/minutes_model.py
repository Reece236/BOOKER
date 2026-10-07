"""
Projected-minutes (rotation) model for the BOOKER forecasts.

The old "projection" was a verbatim clone of last season's *actual* minutes
(`data_ingest/clone_schedule_2027.py`), which baked injuries into the future
(Embiid's 1,200 injury-shortened minutes carried forward) and let deep-bench
garbage-time players drag -- and, when cut, inflate -- team win totals.

This module projects a *healthy* rotation instead:

  1. Role rate (health-free): each player's recency-weighted minutes-per-game
     (mpg) from prior seasons. mpg is the coach's revealed role, independent of
     the player's *rating*: an injured star keeps starter mpg while missing games
     (Embiid ~34 mpg over 39 games), and a negative-rated regular like Jaylen
     Brown still logs ~34 mpg. We deliberately do NOT rank minutes by impact.
  2. Healthy workload: raw = min(mpg, MPG_CAP) * TARGET_GAMES -- restores an
     injured star to a full-season load while leaving a role-limited bench guy low.
  3. Team waterfall to a fixed TEAM_BUDGET (82 * 240 player-minutes): fill players
     in role (mpg) order up to their healthy raw until the budget is spent. This
     yields a realistic ~10-man rotation with a near-zero deep bench, so a
     garbage-time player carries ~no presence and cutting him barely moves wins.

The returned {pid: minutes} is consumed by the team roll-ups (aggregate_net /
aggregate_off_def) with `budget=pi.TEAM_BUDGET`, which additionally charges any
unfilled minutes to a replacement-level player.
"""
from collections import defaultdict

from . import player_impacts as pi

DECAY = pi.DECAY
MPG_CAP = pi.MPG_CAP
TARGET_GAMES = pi.TARGET_GAMES
SEASON_GAMES = pi.SEASON_GAMES
TEAM_BUDGET = pi.TEAM_BUDGET


def _stint_role(data, season):
    """(games, mpg) per normalized name for `season`, from play-by-play stints.

    The box table lags the play-by-play feed by a season, so this is the only role
    signal for the newest draft class (and for everyone's latest season). Games =
    distinct GAME_IDs the player appears in; minutes come from the roster file.
    """
    stints = getattr(data, "STINTS", {}).get(season)
    pl = data.PLAYERS.get(season)
    if stints is None or pl is None or "GAME_ID" not in stints.columns:
        return {}
    games = defaultdict(set)
    for gid, hl, al in zip(stints.GAME_ID, stints.home, stints.away):
        for p in hl:
            games[int(p)].add(gid)
        for p in al:
            games[int(p)].add(gid)
    mins, gp = defaultdict(float), defaultdict(set)
    for pid, name, mn in zip(pl.PLAYER_ID, pl.NAME, pl.MINUTES):
        key = pi.norm_name(name)
        mins[key] += float(mn or 0.0)
        gp[key] |= games.get(int(pid), set())   # traded players: union of games
    return {nm: (len(gp[nm]), mins[nm] / len(gp[nm])) for nm in mins if gp[nm]}


def _weighted_mpg(data, season):
    """Recency-weighted minutes-per-game per normalized name, from seasons < season.

    mpg within each season is weighted by games played (a 70-game season's role is
    more reliable than a 10-game cameo) and by DECAY recency toward `season`. Box
    stats are used where available; play-by-play stints fill player-seasons the
    box table doesn't cover yet.
    """
    role = {(nm, ss): (data.GP.get((nm, ss), 0.0), mpg)
            for (nm, ss), mpg in data.MPG.items() if ss < season}
    for ss in getattr(data, "STINTS", {}):
        if ss >= season:
            continue
        for nm, gm in _stint_role(data, ss).items():
            role.setdefault((nm, ss), gm)

    wnum, wden = {}, {}
    for (nm, ss), (gp, mpg) in role.items():
        if gp <= 0:
            continue
        w = DECAY ** (season - 1 - ss) * gp
        wnum[nm] = wnum.get(nm, 0.0) + w * mpg
        wden[nm] = wden.get(nm, 0.0) + w
    return {nm: wnum[nm] / wden[nm] for nm in wnum if wden[nm] > 0}


def project_minutes(data, season, budget=TEAM_BUDGET, roster=None):
    """Projected healthy rotation minutes {pid: minutes} for `season`'s rosters.

    Minutes are driven by revealed role (mpg), health-restored, and allocated
    within a fixed per-team `budget` so deep-bench players get near-zero share.

    `roster` optionally overrides `data.PLAYERS[season]` (same columns:
    PLAYER_ID, NAME, TEAM_ID, MINUTES) -- e.g. a trade-modified roster -- so the
    rotation is re-projected for the edited teams.
    """
    if roster is None and season not in data.PLAYERS:
        return {}
    mpg_of = _weighted_mpg(data, season)
    pl = roster if roster is not None else data.PLAYERS[season]

    raw, prio, by_team = {}, {}, defaultdict(list)
    for pid, name, tid, obs in zip(pl.PLAYER_ID, pl.NAME, pl.TEAM_ID, pl.MINUTES):
        pid = int(pid)
        by_team[tid].append(pid)
        key = pi.norm_name(name)
        mpg = mpg_of.get(key)
        if mpg and mpg > 0:
            capped = min(mpg, MPG_CAP)
            raw[pid] = capped * TARGET_GAMES
            prio[pid] = capped
        else:
            # no role history (rookie / new arrival): fall back to the observed
            # cloned workload so an unknown can't out-rank established starters.
            obs = float(obs or 0.0)
            raw[pid] = min(obs, MPG_CAP * TARGET_GAMES)
            prio[pid] = min(obs / SEASON_GAMES, MPG_CAP)

    out = {}
    for tid, pids in by_team.items():
        remaining = float(budget)
        # role (mpg) order -- NOT impact rating -- so starters are filled first.
        for pid in sorted(pids, key=lambda p: prio.get(p, 0.0), reverse=True):
            alloc = min(raw.get(pid, 0.0), max(0.0, remaining))
            out[pid] = alloc
            remaining -= alloc
    return out
