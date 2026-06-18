"""
BOOKER in-season win-projection model.

As a season unfolds we update each team's projected final win total by blending
the preseason prior with what's actually happened on the floor. At a series of
cutoff dates we:

  1. rebuild player impacts from prior seasons PLUS the current season's stints
     observed up to that date (the in-season possessions naturally outweigh the
     prior as they accumulate -- the blend the project wanted),
  2. weight each player by minutes-played-to-date to get current team strength,
  3. add wins already banked to the expected wins over the remaining schedule.

Output cache/inseason_timeline_{season}.csv and inseason_timeline_all.csv:
    season, date, frac, team, games_played, wins_to_date, exp_remaining,
    proj_final, pred_net, actual_wins
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import player_impacts as pi

CACHE = pi.CACHE
SEASONS = range(2018, 2027)
N_CUTOFFS = 10            # evenly spaced snapshots through the season


def minutes_to_date(stints, players):
    """Per-player minutes from current-season stints observed so far."""
    mins = {}
    for hl, al, dur in zip(stints.home, stints.away, stints.DURATION_SECONDS):
        for p in hl:
            mins[p] = mins.get(p, 0.0) + dur
        for p in al:
            mins[p] = mins.get(p, 0.0) + dur
    return {p: m / 60.0 for p, m in mins.items()}


def cutoff_dates(reg):
    dates = sorted(reg.DATE.unique())
    if len(dates) < N_CUTOFFS:
        return dates
    qs = np.linspace(1.0 / N_CUTOFFS, 1.0, N_CUTOFFS)
    return [dates[min(len(dates) - 1, int(q * len(dates)) - 0)] for q in qs]


def wins_before(reg, date):
    played = reg[reg.DATE < date]
    w = {}
    for h, a, hw in zip(played.HOME, played.AWAY, played.HOME_WIN):
        win, lose = (h, a) if hw == 1 else (a, h)
        w[win] = w.get(win, 0) + 1
        w.setdefault(lose, 0)
    return w, len(played)


def expected_remaining(reg, date, nets):
    rem = reg[reg.DATE >= date]
    exp = {t: 0.0 for t in nets}
    for h, a in zip(rem.HOME, rem.AWAY):
        if h not in nets or a not in nets:
            continue
        p_home = norm.cdf((nets[h] - nets[a] + pi.HOME_COURT_ADV) / pi.GAME_MARGIN_SD)
        exp[h] += p_home
        exp[a] += 1 - p_home
    return exp


def run_season(data, season):
    if season not in data.GAMES or season not in data.PLAYERS:
        return None
    train = pi.prior_train_seasons(data, season)
    if not train:
        return None
    alpha = pi.pick_alpha(data, train)
    reg = data.GAMES[season]
    reg = reg[reg.get("SEASON_TYPE", "Regular Season") == "Regular Season"].copy()
    abbr = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ABBR))
    season_min = dict(zip(data.PLAYERS[season].PLAYER_ID, data.PLAYERS[season].MINUTES))
    total_dates = sorted(reg.DATE.unique())

    rows = []
    for date in cutoff_dates(reg):
        in_st = data.season_stints_before(season, date)
        if in_st is None:
            in_st = data.STINTS[season].iloc[:0]
        impact, _, last_age = pi.build_impacts(data, train, season, alpha,
                                               extra_stints=in_st, extra_weight=1.0)
        mins = minutes_to_date(in_st, data.PLAYERS[season])
        # fall back to season minutes very early when little has been played
        use_min = mins if sum(mins.values()) > 5000 else season_min
        net_by_tid = pi.aggregate_net(data, impact, season, last_age=last_age,
                                      target_season=season, minutes=use_min)
        nets = {abbr[t]: v for t, v in net_by_tid.items() if t in abbr}
        wtd, gp = wins_before(reg, date)
        exp_rem = expected_remaining(reg, date, nets)
        frac = round(total_dates.index(date) / max(1, len(total_dates) - 1), 3) \
            if date in total_dates else 1.0
        for t in nets:
            wd = wtd.get(t, 0)
            er = exp_rem.get(t, 0.0)
            rows.append({
                "season": season, "date": date, "frac": frac, "team": t,
                "games_played": wd + _losses(reg, date, t),
                "wins_to_date": wd,
                "exp_remaining": round(er, 1),
                "proj_final": round(wd + er, 1),
                "pred_net": round(nets[t], 2),
                "actual_wins": (None if data.ACTUAL_WINS.get((t, season)) is None
                                else int(data.ACTUAL_WINS[(t, season)])),
            })
    return pd.DataFrame(rows)


def _losses(reg, date, team):
    played = reg[reg.DATE < date]
    return int(((played.HOME == team) & (played.HOME_WIN == 0)).sum()
               + ((played.AWAY == team) & (played.HOME_WIN == 1)).sum())


def main():
    data = pi.BookerData()
    frames = []
    for s in SEASONS:
        df = run_season(data, s)
        if df is None:
            print(f"season {s}: skipped")
            continue
        df.to_csv(CACHE / f"inseason_timeline_{s}.csv", index=False)
        final = df[df.frac == df.frac.max()]
        if final.actual_wins.notna().any():
            err = (final.proj_final - final.actual_wins).abs().mean()
            print(f"season {s}: {df.date.nunique()} snapshots, end-of-year "
                  f"proj MAE {err:.1f}")
        else:
            print(f"season {s}: {df.date.nunique()} snapshots (no actuals)")
        frames.append(df)
    allt = pd.concat(frames, ignore_index=True)
    allt.to_csv(CACHE / "inseason_timeline_all.csv", index=False)
    print(f"wrote inseason_timeline_all.csv ({len(allt)} rows)")


if __name__ == "__main__":
    main()
