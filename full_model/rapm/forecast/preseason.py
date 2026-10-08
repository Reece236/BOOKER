"""
BOOKER preseason win-forecast model.

Before a season starts we know each team's roster (and projected minutes) but none
of its results. NO REALIZED MINUTES (2026-10): every team aggregate -- both rating arms,
the uncertainty bands and the playoff top-7 blend -- weights players by PROJECTED
minutes shares on PROJECTED rosters (forecast.minutes_share: opening rotations for past
seasons, ESPN rosters for the live one), never the minutes they went on to play or the
team they finished the season on. Ratings are prior-only: BookerFormer row s-1
(rating through s-1) aged one year, and the enhanced ridge fit on seasons < s. We estimate every player's impact from prior seasons only
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

from . import minutes_share as ms
from . import player_impacts as pi
from . import uncertainty as unc

# NOTE (2026-10 audit): this used to be a single try-block importing enhanced_impacts
# together with bayesian_matchup (PyMC). bayesian_matchup fails to import in the project
# env, so `ei` was ALWAYS None and the "enhanced" arm silently ran the plain ridge path.
# Every validated preseason number was produced by plain ridge -> keep that arm.

CACHE = pi.CACHE
N_SIMS = 4000
SEASONS = range(2018, 2028)


_BK_CACHE = {}


def _booker_aged_nets(data, season):
    """{abbr: net} from the BookerFormer (bible-prior) ratings CSV with ONE year of
    the quadratic age curve applied -- the ratings carry no aging, so a 21-year-old's
    growth and a 35-year-old's decline were ignored (the biggest historical preseason
    misses are exactly these). Returns None if the season isn't in the ratings file."""
    if "df" not in _BK_CACHE:
        if not RATINGS.exists():
            _BK_CACHE["df"] = None
        else:
            r = pd.read_csv(RATINGS)
            a = unc.attach_attrs(r.drop_duplicates("PLAYER_ID", keep="last"))
            _BK_CACHE["df"] = r
            _BK_CACHE["age_ref"] = {int(p): (int(s), float(ag)) for p, s, ag in
                                    zip(a.PLAYER_ID, a.season, a.age) if pd.notna(ag)}
    r = _BK_CACHE["df"]
    # prior-only: the rating THROUGH season-1 (row season-1), aged one year
    if r is None or (season - 1) not in set(r.season):
        return None
    g = r[r.season == season - 1]
    imp = {}
    for pid, v in zip(g.PLAYER_ID, g.impact_total):
        ar = _BK_CACHE["age_ref"].get(int(pid))
        if ar is not None:
            a1 = ar[1] + (season - ar[0])
            v = v + pi.AGE_QUAD * ((a1 - pi.AGE_PEAK) ** 2 - (a1 - 1 - pi.AGE_PEAK) ** 2)
        imp[int(pid)] = float(v)
    return ms.team_nets(imp, season)


def team_nets(data, season):
    """Predicted net rating per team abbreviation for `season` (prior-only).

    ENSEMBLE of two independent engines, averaged 50/50 in net space (OOS 2018-2026:
    schedule-sim wins RMSE 7.52 -> 7.37, better in 7 of 9 seasons): the enhanced-ridge
    path below + age-adjusted BookerFormer ratings (_booker_aged_nets)."""
    train = pi.prior_train_seasons(data, season)
    proj = ms.load(season)
    if not train or proj.empty:
        return None, None, None
    # ridge arm (box-prior ridge RAPM on seasons < s -- the arm that was actually
    # validated, see import note), aged to the target season, aggregated over the
    # PROJECTED roster with PROJECTED minutes shares.
    alpha = pi.pick_alpha(data, train)
    impact, _, last_age = pi.build_impacts(data, train, season, alpha)
    pids = set(proj.PLAYER_ID.astype(int))
    imp = {p: pi.aged_value(impact, p, last_age, season) for p in pids if p in impact}
    nets = ms.team_nets(imp, season, proj=proj)
    bk = _booker_aged_nets(data, season)
    if bk:
        nets = {t: (0.5 * v + 0.5 * bk[t]) if t in bk else v for t, v in nets.items()}
    k, c = pi.fit_net_to_wins(data, train)
    return nets, k, c


def simulate(nets, schedule, net_sd=None, n_sims=N_SIMS, seed=7):
    """Simulate a season's win totals. Two variance sources:
      * within-season binomial -- independent per-game coin flips, and
      * a CORRELATED per-season net-rating shock per team (`net_sd`) that moves ALL
        of a team's ~82 games together (rating uncertainty + roster fragility +
        systematic error).
    The old sim had only the first term, so a team's TRUE net was treated as known and
    the bands covered ~50% of outcomes. The correlated shock is what fattens the
    win-total distribution to the observed ~8.8-win spread (target 80% coverage)."""
    teams = sorted(nets)
    idx = {t: i for i, t in enumerate(teams)}
    sched = schedule[schedule.HOME.isin(idx) & schedule.AWAY.isin(idx)]
    h = sched.HOME.map(idx).to_numpy()
    a = sched.AWAY.map(idx).to_numpy()
    net = np.array([nets[t] for t in teams])

    rng = np.random.default_rng(seed)
    if net_sd:
        seen = [v for v in net_sd.values() if v == v]
        fill = float(np.mean(seen)) if seen else 0.0
        nsd = np.array([net_sd.get(t, fill) for t in teams])
    else:
        nsd = np.zeros(len(teams))
    # per-sim team net (one correlated shock per team-season), then per-sim per-game
    # home win prob from that sim's margins.
    z = rng.standard_normal((n_sims, len(teams)))
    net_s = net[None, :] + z * nsd[None, :]                       # [S, T]
    margin = net_s[:, h] - net_s[:, a] + pi.HOME_COURT_ADV        # [S, G]
    p_home = norm.cdf(margin / pi.GAME_MARGIN_SD)
    home_win = rng.random((n_sims, len(h))) < p_home
    wins = np.zeros((n_sims, len(teams)), dtype=np.int32)
    for g in range(len(h)):
        wins[home_win[:, g], h[g]] += 1
        wins[~home_win[:, g], a[g]] += 1
    return teams, wins, net_s


def _series_p(pg):
    """P(win a best-of-7 series | per-game win prob pg). Closed form: win 4 before opp."""
    q = 1.0 - pg
    return pg ** 4 * (1 + 4 * q + 10 * q * q + 20 * q * q * q)


# playoff-series home edge: higher seed hosts 4 of 7 -> a small net bump (pts/100).
SERIES_HCA = 0.18
# per-series matchup/health uncertainty (net pts): a team doesn't play to its exact
# regular-season net every series (matchups, rotations tighten, injuries). Calibrated
# against 179 completed playoff series 2015-2026: better-net team won 57/70/74/89% at
# net gaps 0-2/2-4/4-7/7+, and 4.2 is the log-loss optimum (0.572; sd=0 gives 0.579,
# sd=7 is worse again) -- so favorites' series odds are honest, not amplified.
SERIES_SD = 4.2


# rotations shorten in the playoffs: a team is its top ~7 guys, not its full bench.
# Blend 40% of a top-7-by-minutes roster net into the playoff gap. Validated on 134
# series 2018-2026: closed-form series log-loss 0.601 -> 0.553 (honest per-fold
# weight 0.561), better in 6 of 9 seasons; every K in 6..10 beats the full roster.
PLAYOFF_TOP7_W = 0.4


def top7_adjust(season):
    """{abbr: top7_net - full_net}: rotation shortening, from the PROJECTED minutes
    shares (top 7 by projected share) and prior-only ratings (row season-1). The old
    version ranked by the minutes players actually played that season."""
    if not RATINGS.exists():
        return {}
    r = pd.read_csv(RATINGS)
    imp = dict(zip(r[r.season == season - 1].PLAYER_ID, r[r.season == season - 1].impact_total))
    proj = ms.load(season)
    adj = {}
    for ab, g in proj.groupby("team"):
        g = g.sort_values("proj_pres", ascending=False)
        def net(gg):
            tm = gg.proj_pres.sum()
            v = gg.PLAYER_ID.map(lambda p: imp.get(int(p), pi.PRIOR_BASE))
            return float((v * gg.proj_pres * (5.0 / tm)).sum()) if tm > 0 else 0.0
        adj[ab] = net(g.head(7)) - net(g)
    return adj


def proj_ratings_frame(ratings, season):
    """Pseudo ratings frame for unc.team_net_sd: projected roster + projected minutes,
    rating sd from the prior-only row (season-1); unrated players get the median sd."""
    proj = ms.load(season)
    r = ratings[ratings.season == season - 1].set_index("PLAYER_ID")
    so = proj.PLAYER_ID.map(r.sd_off).fillna(r.sd_off.median() if len(r) else 2.0)
    sd = proj.PLAYER_ID.map(r.sd_def).fillna(r.sd_def.median() if len(r) else 2.0)
    return pd.DataFrame({"season": season, "team": proj.team, "minutes": proj.proj_minutes,
                         "sd_off": so.values, "sd_def": sd.values})


def title_odds(teams, wins, net_s, seed=11, top7=None):
    """P(win the title) per team: seed each sim's teams by conference wins, then run
    the 8-team bracket per conference + Finals as best-of-7 series whose per-game prob
    comes from that sim's (shock-included) net ratings + a higher-seed home edge + a
    per-series matchup shock. Uses the SAME per-sim net draw as the win totals.
    `top7` = {abbr: top7_net - full_net} rotation-shortening adjustment (see above)."""
    conf = np.array([pi.CONFERENCE.get(t, "E") for t in teams])
    ei = np.where(conf == "E")[0]
    wi = np.where(conf == "W")[0]
    n_sims = wins.shape[0]
    rng = np.random.default_rng(seed)
    champ = np.zeros(len(teams))
    sd = pi.GAME_MARGIN_SD
    BRACKET = [(0, 7), (3, 4), (2, 5), (1, 6)]     # 1-8 seeding (keeps 1 & 2 apart)
    adj = np.array([PLAYOFF_TOP7_W * (top7 or {}).get(t, 0.0) for t in teams])

    def play(a, b):                                 # higher seed = a; returns winner idx
        shock = rng.normal(0.0, SERIES_SD)          # per-series matchup/health variance
        gap = (net_s[s, a] + adj[a]) - (net_s[s, b] + adj[b])
        pg = norm.cdf((gap + SERIES_HCA + shock) / sd)
        return a if rng.random() < _series_p(pg) else b

    def hi(x, y):   # HCA = better record, tie -> net (actual NBA rule, incl. Finals)
        return ((x, y) if (wins[s, x], net_s[s, x]) >= (wins[s, y], net_s[s, y])
                else (y, x))

    for s in range(n_sims):
        finalists = []
        for cols in (ei, wi):
            if len(cols) < 8:
                continue
            seeds = cols[np.argsort(-wins[s, cols])][:8]     # top 8 by wins this sim
            r8 = [play(seeds[i], seeds[j]) for i, j in BRACKET]
            r4 = [play(*hi(r8[0], r8[1])), play(*hi(r8[2], r8[3]))]
            finalists.append(play(*hi(r4[0], r4[1])))
        if len(finalists) == 2:
            champ[play(*hi(finalists[0], finalists[1]))] += 1
    return champ / n_sims


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


RATINGS = CACHE.parent / "booker_bookerformer_ratings.csv"
TARGET_COVERAGE = 0.80


def run_season(data, season, ratings=None, net_scale=1.0, precomp=None):
    nets, k, c = precomp if precomp is not None else team_nets(data, season)
    if nets is None or season not in data.GAMES:
        return None
    sched = data.GAMES[season]
    sched = sched[sched.get("SEASON_TYPE", "Regular Season") == "Regular Season"]
    net_sd = (unc.team_net_sd(proj_ratings_frame(ratings, season), season, net_scale=net_scale)
              if ratings is not None else None)
    teams, wins, net_s = simulate(nets, sched, net_sd=net_sd)
    p_playoff = playoff_odds(teams, wins)
    p_champ = title_odds(teams, wins, net_s, top7=top7_adjust(season))
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
            "p_champ": round(float(p_champ[i]), 4),
            "actual_wins": (None if actual[t] is None else int(actual[t])),
        })
    return pd.DataFrame(rows)


def _coverage(cache, ratings, scale):
    """Pooled fraction of historical team-seasons whose actual wins land in [p10,p90]."""
    hit = tot = 0
    for s, (nets, k, c, sched, actual) in cache.items():
        net_sd = unc.team_net_sd(proj_ratings_frame(ratings, s), s, net_scale=scale)
        teams, wins, _ = simulate(nets, sched, net_sd=net_sd)
        for i, t in enumerate(teams):
            aw = actual.get(t)
            if aw is None:
                continue
            p10, p90 = np.percentile(wins[:, i], [10, 90])
            hit += int(p10 <= aw <= p90); tot += 1
    return hit / tot if tot else float("nan")


def calibrate_net_scale(cache, ratings, lo=0.0, hi=6.0, iters=14):
    """Bisection on the global net-shock scale to hit TARGET_COVERAGE. Coverage is
    monotone increasing in scale (deterministic sims, fixed seed), so this converges
    on the smallest scale whose 80% band covers 80% of outcomes."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _coverage(cache, ratings, mid) < TARGET_COVERAGE:
            lo = mid          # bands too narrow -> widen
        else:
            hi = mid
    return (lo + hi) / 2


def _shrink_predictions(cache):
    """Walk-forward regression-dilution fix, applied to each season's nets in place.

    The net->wins map (k, c) is learned on REALIZED team nets; applying it to
    PREDICTED nets overshoots at the tails because predictions carry error and
    actual outcomes regress: 2018-2025, teams projected for a .67+ win pct
    under-delivered by ~7 wins even in 82-game seasons (slope of actual on
    projected win pct = 0.79). Fit actual_pct ~ proj_pct on completed PRIOR
    seasons only and shrink each season's nets by the equivalent linear map:
        net' = beta*net + (82*alpha - (1-beta)*c) / k
    This is the fix for 60%+ title favorites: the sim's playoff math is honest
    (SERIES_SD validated on real series), the inflated inputs were not.
    """
    hist = []                                    # (season, proj_pct, actual_pct)
    for s in sorted(cache):
        nets, k, c, sched, actual = cache[s]
        gp = pd.concat([sched.HOME, sched.AWAY]).value_counts()
        tot = sum(v for v in actual.values() if v is not None)
        if tot < 0.95 * len(sched):              # partial season: never fit on it
            continue
        for t, n in nets.items():
            if actual.get(t) is not None and gp.get(t):
                hist.append((s, (k * n + c) / 82.0, actual[t] / gp[t]))
    for s in sorted(cache):
        prior = [(p, a) for (ss, p, a) in hist if ss < s]
        if len(prior) < 60:
            continue                             # not enough history: leave raw
        beta, alpha = np.polyfit([p for p, _ in prior], [a for _, a in prior], 1)
        nets, k, c, sched, actual = cache[s]
        shr = {t: beta * v + (82.0 * alpha - (1.0 - beta) * c) / k
               for t, v in nets.items()}
        cache[s] = (shr, k, c, sched, actual)
        print(f"  pred-shrink {s}: beta {beta:.3f} alpha {alpha:.3f} "
              f"(fit on {len(prior)} prior team-seasons)")


def main():
    data = pi.BookerData()
    ratings = pd.read_csv(RATINGS) if RATINGS.exists() else None
    if ratings is None:
        print("WARNING: ratings CSV missing; team bands will use binomial-only variance")
    # precompute nets/schedule/actuals per season once (the expensive part), so the
    # calibration scale-search only re-runs the cheap numpy sim.
    cache = {}
    for s in SEASONS:
        nets, k, c = team_nets(data, s)
        if nets is None or s not in data.GAMES:
            print(f"season {s}: skipped (insufficient priors or no schedule)")
            continue
        sched = data.GAMES[s]
        sched = sched[sched.get("SEASON_TYPE", "Regular Season") == "Regular Season"]
        actual = {ab: data.ACTUAL_WINS.get((ab, s)) for ab in nets}
        cache[s] = (nets, k, c, sched, actual)

    _shrink_predictions(cache)

    scale = 1.0
    if ratings is not None:
        base = _coverage(cache, ratings, 0.0)
        scale = calibrate_net_scale(cache, ratings)
        cov = _coverage(cache, ratings, scale)
        print(f"team-band calibration: binomial-only coverage {100*base:.0f}% -> "
              f"net_scale={scale:.3f} gives {100*cov:.0f}% (target {100*TARGET_COVERAGE:.0f}%)")

    frames = []
    for s, (nets, k, c, sched, actual) in cache.items():
        df = run_season(data, s, ratings=ratings, net_scale=scale, precomp=(nets, k, c))
        df.to_csv(CACHE / f"preseason_{s}.csv", index=False)
        have = df.actual_wins.notna()
        if have.any():
            rmse = float(np.sqrt(((df.proj_wins[have] - df.actual_wins[have]) ** 2).mean()))
            print(f"season {s}: {len(df)} teams, proj-wins RMSE {rmse:.1f}, "
                  f"mean sim_sd {df.sim_sd.mean():.1f}")
        else:
            print(f"season {s}: {len(df)} teams (no actuals yet)")
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    allp.to_csv(CACHE / "preseason_all.csv", index=False)
    have = allp[allp.actual_wins.notna()]
    cov = float(((have.p10 <= have.actual_wins) & (have.actual_wins <= have.p90)).mean())
    print(f"wrote preseason_all.csv ({len(allp)} team-seasons); "
          f"pooled band coverage {100*cov:.0f}% over {len(have)} with actuals")


if __name__ == "__main__":
    main()
