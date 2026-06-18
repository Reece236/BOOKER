"""
BOOKER preseason win-forecast model.

Before a season starts we know each team's roster (and projected minutes) but none
of its results. We estimate every player's impact from prior seasons only
(box-prior-blended RAPM, recency-weighted, aged to the target season), roll the
roster up to a predicted team net rating, and:

  * map net -> expected wins with the learned linear map, and
  * Monte-Carlo simulate the full schedule (per-game win prob from the net-rating
    margin + home court) to get a win-total distribution and playoff odds.

Outputs cache/preseason_{season}.csv and a combined cache/preseason_all.csv:
    season, team, pred_net, proj_wins, sim_mean, sim_sd, p10, p50, p90,
    p_playoff, actual_wins
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import player_impacts as pi

try:
    from . import enhanced_impacts as ei
    from .bayesian_matchup import player_impact_dict as bayesian_impacts
except ImportError:
    ei = None
    bayesian_impacts = None

CACHE = pi.CACHE
N_SIMS = 4000
SEASONS = range(2018, 2028)


def team_nets(data, season):
    """Predicted net rating per team abbreviation for `season` (prior-only)."""
    train = pi.prior_train_seasons(data, season)
    if not train or season not in data.PLAYERS:
        return None, None, None
    if ei is not None:
        enh = ei.build_enhanced(data, train, season)
        _, _, net_by_tid = ei.aggregate_off_def(
            data, enh, season, target_season=season)
    elif bayesian_impacts is not None:
        off, def_, tot = bayesian_impacts(season)
        if tot:
            impact = tot
            net_by_tid = pi.aggregate_net(data, impact, season, target_season=season)
        else:
            alpha = pi.pick_alpha(data, train)
            impact, _, last_age = pi.build_impacts(data, train, season, alpha)
            net_by_tid = pi.aggregate_net(data, impact, season,
                                          last_age=last_age, target_season=season)
    else:
        alpha = pi.pick_alpha(data, train)
        impact, _, last_age = pi.build_impacts(data, train, season, alpha)
        net_by_tid = pi.aggregate_net(data, impact, season,
                                      last_age=last_age, target_season=season)
    abbr = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ABBR))
    nets = {abbr[t]: v for t, v in net_by_tid.items() if t in abbr}
    k, c = pi.fit_net_to_wins(data, train)
    return nets, k, c


def simulate(nets, schedule, n_sims=N_SIMS, seed=7):
    """Simulate a season's win totals from per-game net-margin win probabilities."""
    teams = sorted(nets)
    idx = {t: i for i, t in enumerate(teams)}
    sched = schedule[schedule.HOME.isin(idx) & schedule.AWAY.isin(idx)]
    h = sched.HOME.map(idx).to_numpy()
    a = sched.AWAY.map(idx).to_numpy()
    net = np.array([nets[t] for t in teams])
    margin = net[h] - net[a] + pi.HOME_COURT_ADV
    p_home = norm.cdf(margin / pi.GAME_MARGIN_SD)

    rng = np.random.default_rng(seed)
    wins = np.zeros((n_sims, len(teams)), dtype=np.int32)
    draws = rng.random((n_sims, len(h)))
    home_win = draws < p_home
    for g in range(len(h)):
        wins[home_win[:, g], h[g]] += 1
        wins[~home_win[:, g], a[g]] += 1
    return teams, wins


def playoff_odds(teams, wins):
    """P(finish top-8 in conference) per team across simulations."""
    conf = np.array([pi.CONFERENCE.get(t, "E") for t in teams])
    p = np.zeros(len(teams))
    for c in ("E", "W"):
        cols = np.where(conf == c)[0]
        if len(cols) == 0:
            continue
        sub = wins[:, cols]
        # rank within conference each sim; top 8 make it
        order = np.argsort(-sub, axis=1)
        made = np.zeros_like(sub, dtype=bool)
        rows = np.arange(sub.shape[0])[:, None]
        made[rows, order[:, :8]] = True
        p[cols] = made.mean(axis=0)
    return p


def run_season(data, season):
    nets, k, c = team_nets(data, season)
    if nets is None or season not in data.GAMES:
        return None
    sched = data.GAMES[season]
    sched = sched[sched.get("SEASON_TYPE", "Regular Season") == "Regular Season"]
    teams, wins = simulate(nets, sched)
    p_playoff = playoff_odds(teams, wins)
    actual = {ab: data.ACTUAL_WINS.get((ab, season)) for ab in teams}
    rows = []
    for i, t in enumerate(teams):
        w = wins[:, i]
        rows.append({
            "season": season, "team": t,
            "pred_net": round(nets[t], 2),
            "proj_wins": round(k * nets[t] + c, 1),
            "sim_mean": round(w.mean(), 1),
            "sim_sd": round(w.std(), 1),
            "p10": int(np.percentile(w, 10)),
            "p50": int(np.percentile(w, 50)),
            "p90": int(np.percentile(w, 90)),
            "p_playoff": round(float(p_playoff[i]), 3),
            "actual_wins": (None if actual[t] is None else int(actual[t])),
        })
    return pd.DataFrame(rows)


def main():
    data = pi.BookerData()
    frames = []
    for s in SEASONS:
        df = run_season(data, s)
        if df is None:
            print(f"season {s}: skipped (insufficient priors or no schedule)")
            continue
        df.to_csv(CACHE / f"preseason_{s}.csv", index=False)
        have = df.actual_wins.notna()
        if have.any():
            err = (df.proj_wins[have] - df.actual_wins[have])
            rmse = float(np.sqrt((err ** 2).mean()))
            print(f"season {s}: {len(df)} teams, proj-wins RMSE {rmse:.1f} vs actual")
        else:
            print(f"season {s}: {len(df)} teams (no actuals yet)")
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    allp.to_csv(CACHE / "preseason_all.csv", index=False)
    print(f"wrote preseason_all.csv ({len(allp)} team-seasons)")


if __name__ == "__main__":
    main()
