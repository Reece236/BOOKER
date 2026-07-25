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
    MPG, GAMES = {}, {}
    for pid, name, tid, obs, mpg, games in rows:
        key = pi.norm_name(name)
        if mpg is not None:
            MPG[(key, season_prior)] = mpg
            GAMES[(key, season_prior)] = games
    return types.SimpleNamespace(PLAYERS={season: pl}, MPG=MPG, GAMES=GAMES), season


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


if __name__ == "__main__":
    test_injured_star_restored()
    test_minutes_not_sorted_by_rating()
    test_cutting_deep_bench_is_a_non_event()
    test_old_behavior_would_inflate()
    print("\nall checks passed")
