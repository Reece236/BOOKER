"""Run BookerFormer and export SEASON ratings: row `season = s` is the player's rating
THROUGH season s (trained on s-2, s-1, s with recency decay anchored at s+1).

Season-label fix (2026-10): rows used to be labelled with the season they FORECAST
(row s trained on s-3..s-1), so a card labelled 2025-26 had seen none of 2025-26 and
every same-season comparison (vs DARKO season-end, pure RAPM, real +/-) was a season
behind. Now row s = "as of the end of s". The same fit is the PRIOR for season s+1:
any prior-only consumer (preseason sim, backtests, sequential in-season filter) must
read row s-1 for target season s -- never row s.

Team / minutes / WAA on row s are season s's (descriptive). Projected minutes for a
forecast season live in forecast/minutes_share.py, not here.

Usage:
    python build_bookerformer_ratings.py              # all seasons 2017..2026
    python build_bookerformer_ratings.py 2025 2026    # only these seasons
"""
import sys
from pathlib import Path

import pandas as pd

from forecast import player_impacts as pi
from forecast.bookerformer import fit_bookerformer
from stint_off_def import enrich_season

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DEFAULT_SEASONS = list(range(2017, 2027))

# BOOKER score = the player's predictive PLUS-MINUS PER 100 POSSESSIONS given average
# starting-caliber teammates and opponents (the regularized model's context-isolated
# impact -- his effect on net rating when dropped into an average-starters game).
# ROLE_LAMBDA (usage regressed toward the skill-curve optimum) is RETIRED at 0:
# the 2026-07-04 audit showed the re-role credit has NEGATIVE forward value on
# both neutral arbiters (arb ~ impact + upside: implied lambda -1.3, t=-2.1 vs
# pure RAPM t+1, t=-1.9 vs real +/-), and high-upside players skew low-usage --
# so the old +0.25 term systematically inflated role players ("misused" guys do
# NOT cash their theoretical upside). The optimal-usage read stays on the player
# card via the role panel; it just no longer buys rating points.
ROLE_LAMBDA = 0.0


def main(seasons):
    for s in range(2015, 2028):
        if (CACHE / f"stints_{s}.csv").exists():
            enrich_season(s)

    data = pi.BookerData(seasons=range(2015, 2028))
    # usage-optimum upside per (season, pid) for the BOOKER usage-regression term;
    # future/projection seasons carry each player's latest available upside forward.
    rvp = CACHE / "role_value.csv"
    upside, up_latest = {}, {}
    if rvp.exists():
        for r in pd.read_csv(rvp).sort_values("season").itertuples():
            upside[(int(r.season), int(r.PLAYER_ID))] = float(r.role_upside)
            up_latest[int(r.PLAYER_ID)] = float(r.role_upside)

    def _up(season, pid):
        return upside.get((season, pid), up_latest.get(pid, 0.0))

    all_rows = []
    for target in seasons:
        # rating THROUGH `target`: train on target-2..target, decay anchored at target+1
        train = [s for s in range(target - 2, target + 1) if s in data.STINTS]
        if len(train) < 2 or target not in data.STINTS:
            print(f"skip {target}: needs observed stints for {target} and >=2 training seasons")
            continue
        print(f"=== fitting BookerFormer through {target} (train {train}) ===")
        post, _ = fit_bookerformer(train, target + 1, data, verbose=True)
        pl = data.PLAYERS.get(target)
        if pl is None:
            continue
        tmin = pl.groupby("TEAM_ID").MINUTES.transform("sum")
        pl = pl.copy()
        pl["presence"] = pl.MINUTES / (tmin / 5.0)
        k, _ = pi.fit_net_to_wins(data, train)
        for r in post.itertuples():
            row = pl[pl.PLAYER_ID == r.PLAYER_ID]
            if row.empty:
                continue
            pres = float(row.presence.iloc[0])
            all_rows.append({
                "season": target,
                "rating_through": target,
                "PLAYER_ID": int(r.PLAYER_ID),
                "player": r.NAME,
                "team": data.abbr_of.get(int(row.TEAM_ID.iloc[0]), "?"),
                "minutes": int(row.MINUTES.iloc[0]),
                "impact_off": round(r.impact_off, 2),
                "impact_def": round(r.impact_def, 2),
                "impact_total": round(r.impact_total, 2),
                "sd_off": round(r.sd_off, 3),
                "sd_def": round(r.sd_def, 3),
                "waa_off": round(k * r.impact_off * pres, 2),
                "waa_def": round(k * r.impact_def * pres, 2),
                "waa_total": round(k * r.impact_total * pres, 2),
                # BOOKER: predictive +/- per 100 poss, avg starting-caliber context,
                # usage regressed toward optimum (offense carries the role term)
                "booker_score": round(r.impact_total + ROLE_LAMBDA * _up(target, int(r.PLAYER_ID)), 2),
                "booker_off": round(r.impact_off + ROLE_LAMBDA * _up(target, int(r.PLAYER_ID)), 2),
                "booker_def": round(r.impact_def, 2),
            })
    if not all_rows:
        print("no rows produced")
        return
    out = pd.DataFrame(all_rows)
    out = out[out.minutes >= 250].copy()
    # Heteroscedastic uncertainty: overwrite the (near-constant) model sd with a
    # per-player posterior std that varies with minutes/age/position/usage. Point
    # estimates (impact/waa/booker_score) are left exactly as the model produced.
    from forecast import uncertainty as unc
    _u = unc.attach_attrs(out)
    so, sd = unc.player_sd(_u)
    out["sd_off"] = so.round(3)
    out["sd_def"] = sd.round(3)
    path = HERE / "booker_bookerformer_ratings.csv"
    # merge-don't-clobber: running a subset of seasons keeps previously built rows
    if path.exists():
        old = pd.read_csv(path)
        # rows without `rating_through` use the retired forecast-season labelling
        old = old[~old.season.isin(set(out.season))] if "rating_through" in old.columns \
            else old.iloc[:0]
        out = pd.concat([old, out], ignore_index=True).sort_values(["season", "PLAYER_ID"])
    out["rank"] = out.groupby("season").waa_total.rank(ascending=False, method="first").astype(int)
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or DEFAULT_SEASONS
    main(seasons)
