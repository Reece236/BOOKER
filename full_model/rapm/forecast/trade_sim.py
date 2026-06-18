"""
Simulate a trade: swap players between teams and estimate wins added/lost.

Uses enhanced impacts (teammate-fit adjusted) and the preseason Monte Carlo
engine to translate net-rating changes into projected win totals.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import contract_value as cv
from . import enhanced_impacts as ei
from . import player_impacts as pi
from .preseason import simulate, playoff_odds

CACHE = pi.CACHE
MAX_TRADE_ASSETS = 6


@dataclass
class TradeResult:
    season: int
    team_a: str
    team_b: str
    player_a: str
    player_b: str
    wins_a_before: float
    wins_a_after: float
    wins_b_before: float
    wins_b_after: float
    net_a_before: float
    net_a_after: float
    net_b_before: float
    net_b_after: float
    delta_wins_a: float
    delta_wins_b: float


def _tid_of(data, season, abbr):
    teams = data.TEAMS[season]
    m = dict(zip(teams.ABBR, teams.TEAM_ID))
    return int(m[abbr])


def _abbr_nets(net_by_tid, data, season):
    abbr = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ABBR))
    return {abbr[t]: v for t, v in net_by_tid.items() if t in abbr}


def _proj_wins(nets, k, c):
    return {t: k * v + c for t, v in nets.items()}


def _clone_roster(data, season):
    pl = data.PLAYERS[season].copy()
    return pl


def apply_trade(pl, team_a, team_b, pid_a, pid_b, data, season):
    """Return new players DataFrame with swapped team assignments."""
    return apply_multi_trade(pl, team_a, team_b, [pid_a], [pid_b], data, season)


def apply_multi_trade(pl, team_a, team_b, pids_a, pids_b, data, season):
    """Move up to MAX_TRADE_ASSETS players from each team to the other."""
    out = pl.copy()
    ta, tb = _tid_of(data, season, team_a), _tid_of(data, season, team_b)
    for pid in pids_a[:MAX_TRADE_ASSETS]:
        out.loc[out.PLAYER_ID == pid, "TEAM_ID"] = tb
    for pid in pids_b[:MAX_TRADE_ASSETS]:
        out.loc[out.PLAYER_ID == pid, "TEAM_ID"] = ta
    return out


def salary_match_check(salary_out, salary_in):
    """
    Simplified CBA salary matching (non-taxpayer, over-cap trade band).
    Incoming must be within 125% + $250k of outgoing when outgoing is larger.
    """
    out = float(sum(salary_out))
    inn = float(sum(salary_in))
    if out <= 0 and inn <= 0:
        return True, "no salary movement"
    if out <= inn:
        return True, "incoming covers outgoing"
    allowed = 1.25 * out + 250_000
    ok = inn >= allowed - 1
    return ok, f"need ${max(0, allowed - inn):,.0f} more incoming" if not ok else "matched"


def roster_minutes(pl):
    return dict(zip(pl.PLAYER_ID.astype(int), pl.MINUTES.astype(float)))


def nets_from_roster(data, season, pl, enh):
    mins = roster_minutes(pl)
    _, _, tot = ei.aggregate_off_def(data, enh, season, target_season=season, minutes=mins)
    return _abbr_nets(tot, data, season)


def simulate_trade(data, season, team_a, pid_a, team_b, pid_b,
                   enh=None, n_sims=4000):
    """Compute before/after projected wins for a 1-for-1 trade."""
    if season not in data.GAMES or season not in data.PLAYERS:
        raise ValueError(f"season {season} missing roster or schedule")

    train = pi.prior_train_seasons(data, season)
    if enh is None:
        enh = ei.build_enhanced(data, train, season)
    k, c = pi.fit_net_to_wins(data, train)

    pl = _clone_roster(data, season)
    name_a = pl.loc[pl.PLAYER_ID == pid_a, "NAME"].iloc[0]
    name_b = pl.loc[pl.PLAYER_ID == pid_b, "NAME"].iloc[0]

    before = nets_from_roster(data, season, pl, enh)
    proj_before = _proj_wins(before, k, c)

    pl2 = apply_trade(pl, team_a, team_b, pid_a, pid_b, data, season)
    after = nets_from_roster(data, season, pl2, enh)
    proj_after = _proj_wins(after, k, c)

    sched = data.GAMES[season]
    sched = sched[sched.get("SEASON_TYPE", "Regular Season") == "Regular Season"]

    teams_b, wins_b = simulate(before, sched, n_sims=n_sims)
    teams_a, wins_a = simulate(after, sched, n_sims=n_sims)
    idx_b = {t: i for i, t in enumerate(teams_b)}
    idx_a = {t: i for i, t in enumerate(teams_a)}

    return TradeResult(
        season=season,
        team_a=team_a, team_b=team_b,
        player_a=name_a, player_b=name_b,
        wins_a_before=float(wins_b[:, idx_b[team_a]].mean()),
        wins_a_after=float(wins_a[:, idx_a[team_a]].mean()),
        wins_b_before=float(wins_b[:, idx_b[team_b]].mean()),
        wins_b_after=float(wins_a[:, idx_a[team_b]].mean()),
        net_a_before=before[team_a], net_a_after=after[team_a],
        net_b_before=before[team_b], net_b_after=after[team_b],
        delta_wins_a=float(wins_a[:, idx_a[team_a]].mean() - wins_b[:, idx_b[team_a]].mean()),
        delta_wins_b=float(wins_a[:, idx_a[team_b]].mean() - wins_b[:, idx_b[team_b]].mean()),
    )


def roster_options(data, season, min_minutes=200):
    """Players eligible for trade UI."""
    pl = data.PLAYERS[season]
    teams = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ABBR))
    rows = []
    for r in pl.itertuples():
        if r.MINUTES < min_minutes:
            continue
        rows.append({
            "pid": int(r.PLAYER_ID),
            "player": str(r.NAME),
            "team": teams.get(r.TEAM_ID, "?"),
            "minutes": int(r.MINUTES),
        })
    return sorted(rows, key=lambda x: (-x["minutes"], x["player"]))


def _player_ages_for_season(season):
    ages, latest = cv._player_ages()
    out = {}
    for (nm, ss), age in ages.items():
        if ss == season:
            out[nm] = age
    for nm in latest.index:
        out.setdefault(nm, float(latest.loc[nm, "age"]))
    return out


def _years_pro_map():
    sal = cv.load_salary_history()
    cnt = sal.groupby("nm").season.nunique()
    return {nm: int(v) for nm, v in cnt.items()}


def export_trade_payload(data, season=2027):
    """Build JSON-serializable trade simulator inputs for the dashboard."""
    train = pi.prior_train_seasons(data, season)
    if not train:
        season = max(s for s in data.seasons if s in data.PLAYERS)
        train = pi.prior_train_seasons(data, season)
    enh = ei.build_enhanced(data, train, season)
    k, c = pi.fit_net_to_wins(data, train)
    comps = ei.player_waa_components(data, season, k, enh)
    roster = roster_options(data, season)
    teams = sorted({r["team"] for r in roster})
    waa_map = cv.build_waa_name_map(data, season)
    cv.fit_model(waa_map)
    ages = _player_ages_for_season(season)
    yos = _years_pro_map()
    return {
        "season": season,
        "teams": teams,
        "roster": roster,
        "components": comps,
        "maxAssets": MAX_TRADE_ASSETS,
        "capRules": cv.cap_rules_payload(),
        "inflation": cv.inflation_table(),
        "waaMap": waa_map,
        "ages": ages,
        "yearsPro": yos,
    }


def main():
    data = pi.BookerData(seasons=range(2015, 2028))
    season = 2027 if 2027 in data.GAMES else 2026
    pl = data.PLAYERS[season]
    teams = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ABBR))
    # demo: top minute guy on LAL for top on NOP if exists
    lal = pl[pl.TEAM_ID.map(lambda t: teams.get(t)) == "LAL"].sort_values("MINUTES", ascending=False)
    nop = pl[pl.TEAM_ID.map(lambda t: teams.get(t)) == "NOP"].sort_values("MINUTES", ascending=False)
    if len(lal) and len(nop):
        r = simulate_trade(data, season, "LAL", int(lal.iloc[0].PLAYER_ID),
                           "NOP", int(nop.iloc[0].PLAYER_ID))
        print(f"Demo trade {season}: {r.player_a} ({r.team_a}) for {r.player_b} ({r.team_b})")
        print(f"  {r.team_a}: {r.wins_a_before:.1f} -> {r.wins_a_after:.1f} W ({r.delta_wins_a:+.1f})")
        print(f"  {r.team_b}: {r.wins_b_before:.1f} -> {r.wins_b_after:.1f} W ({r.delta_wins_b:+.1f})")


if __name__ == "__main__":
    main()
