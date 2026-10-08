"""
Unit checks for the projected-minutes / rotation model and the budget-aware
team roll-up. Uses lightweight synthetic rosters, so it runs without the large
`cache/` artifacts.

Run from the `full_model/rapm` directory:
    python -m forecast.test_minutes_model
"""
import types

import pandas as pd

from . import minutes_model as mm
from . import player_impacts as pi


def _fake_data(rows, season_prior=2026):
    """rows: list of (pid, name, tid, obs_min, mpg, games). Builds a stub that
    exposes just what project_minutes / aggregate_net read."""
    season = season_prior + 1
    pl = pd.DataFrame([{
        "PLAYER_ID": r[0], "NAME": r[1], "TEAM_ID": r[2], "MINUTES": r[3],
    } for r in rows])
    MPG, GP = {}, {}
    for pid, name, tid, obs, mpg, games in rows:
        key = pi.norm_name(name)
        if mpg is not None:
            MPG[(key, season_prior)] = mpg
            GP[(key, season_prior)] = games
    return types.SimpleNamespace(PLAYERS={season: pl}, MPG=MPG, GP=GP), season


def test_injured_star_restored():
    # Embiid-shaped: 33.6 mpg but only 39 games -> observed ~1,310 min.
    data, season = _fake_data([
        (1, "Joel Embiid", 10, 1310, 33.6, 39),
        (2, "Starter B", 10, 2200, 30.0, 74),
        (3, "Starter C", 10, 2000, 28.0, 74),
        (4, "Starter D", 10, 1800, 25.0, 74),
        (5, "Starter E", 10, 1700, 24.0, 74),
    ])
    mins = mm.project_minutes(data, season)
    assert 2300 <= mins[1] <= 2500, f"Embiid projected {mins[1]:.0f}, expected ~2400"
    print(f"[ok] injured star restored: Embiid 1310 obs -> {mins[1]:.0f} projected")


def test_minutes_not_sorted_by_rating():
    # A negative-value 34-mpg starter must still get starter minutes: the minutes
    # model keys off role (mpg), never the impact rating.
    data, season = _fake_data([
        (1, "Jaylen Brown", 20, 2158, 34.0, 63),   # would be low if sorted by rating
        (2, "Scrub A", 20, 300, 8.0, 40),
        (3, "Scrub B", 20, 300, 8.0, 40),
        (4, "Scrub C", 20, 300, 8.0, 40),
        (5, "Scrub D", 20, 300, 8.0, 40),
    ])
    mins = mm.project_minutes(data, season)
    assert mins[1] > 2000, f"negative-rated starter got {mins[1]:.0f}, expected >2000"
    assert mins[1] == max(mins.values()), "starter should lead the team in minutes"
    print(f"[ok] role over rating: negative-value 34-mpg starter -> {mins[1]:.0f} min")


def _okc_like():
    # 5 starters + bench + two garbage-time rookies (Topic / Barnhizer shapes).
    rows = [
        (1, "SGA", 30, 2598, 34.2, 76),
        (2, "Holmgren", 30, 2100, 32.0, 66),
        (3, "Dort", 30, 2160, 30.0, 72),
        (4, "Wallace", 30, 2016, 28.0, 72),
        (5, "Hartenstein", 30, 1900, 27.0, 70),
        (6, "Caruso", 30, 1600, 23.0, 68),
        (7, "Wiggins", 30, 1300, 20.0, 65),
        (8, "Joe", 30, 1200, 18.0, 66),
        (9, "Topic", 30, 649, 14.4, 45),      # garbage-time rookie, impact -2.9
        (10, "Barnhizer", 30, 694, 14.5, 48),  # garbage-time rookie, impact -2.7
    ]
    impact = {1: 11.3, 2: 4.9, 3: 1.0, 4: 1.0, 5: 2.0,
              6: 3.0, 7: 0.0, 8: 0.0, 9: -2.9, 10: -2.7}
    return rows, impact


def _team_net(rows, impact, budget):
    data, season = _fake_data(rows)
    mins = mm.project_minutes(data, season, budget=budget) if budget else None
    net = pi.aggregate_net(data, impact, season, minutes=mins, budget=budget)
    return net[30]


def test_cutting_deep_bench_is_a_non_event():
    rows, impact = _okc_like()
    budget = pi.TEAM_BUDGET
    base = _team_net(rows, impact, budget)
    # cut the two garbage-time rookies
    cut = [r for r in rows if r[0] not in (9, 10)]
    after_bench = _team_net(cut, impact, budget)
    # cut a star instead
    cut_star = [r for r in rows if r[0] != 1]
    after_star = _team_net(cut_star, impact, budget)

    d_bench = after_bench - base
    d_star = after_star - base
    k = 2.5  # approx net->wins slope
    print(f"[info] base net {base:.2f}; cut bench -> {after_bench:.2f} "
          f"(Δ{d_bench:+.2f} net, ~{k*d_bench:+.1f} wins); "
          f"cut SGA -> {after_star:.2f} (Δ{d_star:+.2f} net, ~{k*d_star:+.1f} wins)")
    assert abs(k * d_bench) < 0.7, f"cutting deep bench moved wins {k*d_bench:+.1f} (should be ~0)"
    assert d_star < -1.0, f"cutting a star should hurt; got Δnet {d_star:+.2f}"
    print("[ok] deep-bench cut ~0 wins; star cut hurts")


def test_old_behavior_would_inflate():
    # Demonstrate the fixed formula beats the legacy own-sum normalization, which
    # RAISES the total when you cut negative garbage-time players.
    rows, impact = _okc_like()
    base_old = _team_net(rows, impact, budget=None)
    cut = [r for r in rows if r[0] not in (9, 10)]
    after_old = _team_net(cut, impact, budget=None)
    print(f"[info] legacy own-sum: cut bench Δnet {after_old - base_old:+.2f} "
          f"(the old bug: cutting scrubs raised the rating)")
    assert after_old > base_old, "sanity: legacy formula inflates on a bench cut"
    print("[ok] reproduced the legacy inflation the fix removes")


# ---------------------------------------------------------------------------
# End-to-end checks on the *real* BookerData constructor, fed a tiny temporary
# cache. One season of play-by-play (2026) that the box table doesn't cover, a
# rookie the box table has never seen, and a veteran whose box ages stop at 2025.
# ---------------------------------------------------------------------------
VET = "Jaylen Brown"        # in the box table through 2025
ROOKIE = "Test Sophomore"   # not in the box table at all (2025-26 rookie)


def _tiny_real_data(tmp):
    from pathlib import Path
    tmp = Path(tmp)
    ids = {VET: 1, ROOKIE: 2, "Filler A": 3, "Filler B": 4, "Filler C": 5}
    away = "11, 12, 13, 14, 15"
    stints = []
    for g in range(60):                       # vet plays 60 games, rookie 50
        home = [1, 3, 4, 5] + ([2] if g < 50 else [6])
        stints.append({"GAME_ID": 1000 + g, "PERIOD": 1,
                       "HOME_LINEUP": ", ".join(map(str, home)), "AWAY_LINEUP": away,
                       "POSS": 50, "Y": 0.0})
    pd.DataFrame(stints).to_csv(tmp / "stints_2026.csv", index=False)
    roster = pd.DataFrame([{"PLAYER_ID": pid, "NAME": nm, "TEAM_ID": 99,
                            "MINUTES": {1: 2100, 2: 1100}.get(pid, 1500)}
                           for nm, pid in ids.items()])
    for s in (2026, 2027):                    # 2027 = cloned projection roster
        roster.to_csv(tmp / f"players_{s}.csv", index=False)
        pd.DataFrame([{"TEAM_ID": 99, "ABBR": "TST"}]).to_csv(
            tmp / f"teams_{s}.csv", index=False)
        pd.DataFrame([{"GAME_ID": 1, "DATE": f"{s}-01-01", "HOME": "TST", "AWAY": "OPP",
                       "HOME_WIN": 1, "SEASON_TYPE": "Regular Season"}]).to_csv(
            tmp / f"games_{s}.csv", index=False)
    pd.DataFrame(columns=["team_abbr", "season", "actual_wins", "predicted_wins"]).to_csv(
        tmp / "team_predictions.csv", index=False)

    old = (pi.CACHE, pi.TEAM_PRED)
    pi.CACHE, pi.TEAM_PRED = tmp, tmp / "team_predictions.csv"
    try:
        data = pi.BookerData(seasons=range(2026, 2028))
    finally:
        pi.CACHE, pi.TEAM_PRED = old
    return data, ids


def test_real_data_end_to_end():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, ids = _tiny_real_data(tmp)

    # the season schedules must survive (regression: GAMES was once clobbered by
    # the games-played dict, silently breaking preseason/trade/export)
    assert isinstance(data.GAMES.get(2026), pd.DataFrame), "schedule dict was overwritten"
    print("[ok] season schedules intact (data.GAMES)")

    rk, vk = pi.norm_name(ROOKIE), pi.norm_name(VET)
    # newcomer: ROOKIE_AGE in his first season, +1 the next
    assert data.AGE[(rk, 2026)] == pi.ROOKIE_AGE and data.AGE[(rk, 2027)] == pi.ROOKIE_AGE + 1
    # veteran: extrapolated from his last box age
    assert abs(data.AGE[(vk, 2026)] - (data.AGE[(vk, 2025)] + 1)) < 1e-9
    print(f"[ok] ages backfilled: sophomore {data.AGE[(rk, 2027)]:.0f} in 2027, "
          f"vet {data.AGE[(vk, 2025)]:.1f} -> {data.AGE[(vk, 2026)]:.1f}")

    # sophomore role rate comes from stints: 1,100 min / 50 games = 22 mpg
    mins = mm.project_minutes(data, 2027)
    expect = 22.0 * pi.TARGET_GAMES
    assert abs(mins[ids[ROOKIE]] - expect) < 1.0, f"sophomore {mins[ids[ROOKIE]]:.0f} vs {expect:.0f}"
    print(f"[ok] sophomore projected from stint mpg: {mins[ids[ROOKIE]]:.0f} min "
          f"(raw fallback would have been 1100)")

    # sophomore gets the young-player aging bump from his 2026 season
    _, last_age = pi._decayed_priors(data, [2026], 2027)
    assert last_age[ids[ROOKIE]] == (2026, pi.ROOKIE_AGE)
    bump = pi.aged_value({ids[ROOKIE]: 0.0}, ids[ROOKIE], last_age, 2027)
    assert bump > 0.3, f"aging bump {bump:.2f}"
    print(f"[ok] sophomore aged from his rookie season: +{bump:.2f}/100 growth")
    return data, ids, mins


def test_projected_presence_keeps_observed_credibility():
    from . import enhanced_impacts as ei
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data, ids = _tiny_real_data(tmp)
    pids = list(ids.values())
    enh = types.SimpleNamespace(
        off={p: 1.5 for p in pids}, def_={p: 1.0 for p in pids},
        total={p: 2.5 for p in pids}, prior={p: -1.0 for p in pids}, last_age={})
    k = 100.0   # large k so 2-decimal rounding can't mask a difference
    obs = {r["pid"]: r for r in ei.player_waa_components(data, 2027, k, enh)}
    proj_min = {p: 2000.0 for p in pids}
    prj = {r["pid"]: r for r in ei.player_waa_components(
        data, 2027, k, enh, proj_minutes=proj_min, budget=pi.TEAM_BUDGET)}
    tmin = sum(float(m) for m in data.PLAYERS[2027].MINUTES)
    for p in pids:
        rate_obs = obs[p]["waa_total"] / (k * obs[p]["minutes"] / (tmin / 5.0))
        rate_prj = prj[p]["waa_total"] / (k * 2000.0 / (pi.TEAM_BUDGET / 5.0))
        assert abs(rate_obs - rate_prj) < 0.01, (p, rate_obs, rate_prj)
        assert prj[p]["minutes"] == 2000.0
    print("[ok] projected presence: same shrunk rate (credibility on observed minutes)")


if __name__ == "__main__":
    test_injured_star_restored()
    test_minutes_not_sorted_by_rating()
    test_cutting_deep_bench_is_a_non_event()
    test_old_behavior_would_inflate()
    test_real_data_end_to_end()
    test_projected_presence_keeps_observed_credibility()
    print("\nall checks passed")
