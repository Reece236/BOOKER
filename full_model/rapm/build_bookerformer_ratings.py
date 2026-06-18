"""Run the BookerFormer Bayesian transformer and export season ratings for the dashboard.

Mirrors build_bayesian_ratings.py exactly -- same training window (prior 3 seasons),
same WAA construction (k * impact * presence) and minutes filter -- but swaps the
PyMC matchup model for forecast.bookerformer.fit_bookerformer. Output schema is
identical to booker_bayesian_ratings.csv so export_dashboard_data.py and any
downstream consumer can read it unchanged.

Usage:
    python build_bookerformer_ratings.py              # all seasons 2018..2027
    python build_bookerformer_ratings.py 2024 2025    # only these target seasons
"""
import sys
from pathlib import Path

import pandas as pd

from forecast import player_impacts as pi
from forecast.bookerformer import fit_bookerformer
from stint_off_def import enrich_season

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DEFAULT_SEASONS = list(range(2018, 2028))

# BOOKER score = forward-looking, skill-based predictive WAA per 3000 possessions
# (no age effect). A player at presence 1.0 (every minute, 48x82) logs ~8200
# possessions, so 3000 poss is the fraction 3000/8200 of a full-time season.
BOOKER_POSS = 3000.0
FULL_SEASON_POSS = 8200.0   # 48 min x 82 games at 100 poss / 48 min


def main(seasons):
    for s in range(2015, 2028):
        if (CACHE / f"stints_{s}.csv").exists():
            enrich_season(s)

    data = pi.BookerData(seasons=range(2015, 2028))
    all_rows = []
    for target in seasons:
        train = [s for s in range(target - 3, target) if s in data.STINTS]
        if len(train) < 2:
            print(f"skip {target}: only {len(train)} training seasons")
            continue
        print(f"=== fitting BookerFormer for {target} (train {train}) ===")
        post, _ = fit_bookerformer(train, target, data, verbose=True)
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
                # BOOKER score: predictive WAA per 3000 possessions at this skill level
                "booker_score": round(k * r.impact_total * (BOOKER_POSS / FULL_SEASON_POSS), 2),
                "booker_off": round(k * r.impact_off * (BOOKER_POSS / FULL_SEASON_POSS), 2),
                "booker_def": round(k * r.impact_def * (BOOKER_POSS / FULL_SEASON_POSS), 2),
            })
    if not all_rows:
        print("no rows produced")
        return
    out = pd.DataFrame(all_rows)
    out = out[out.minutes >= 250].copy()
    out["rank"] = out.groupby("season").waa_total.rank(ascending=False, method="first").astype(int)
    path = HERE / "booker_bookerformer_ratings.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or DEFAULT_SEASONS
    main(seasons)
