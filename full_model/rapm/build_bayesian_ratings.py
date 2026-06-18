"""Run PyMC Bayesian matchup model and export season ratings for the dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import player_impacts as pi
from forecast.bayesian_matchup import fit_bayesian, OUT
from stint_off_def import enrich_season

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SEASONS = list(range(2018, 2028))


def main():
    for s in SEASONS:
        if (CACHE / f"stints_{s}.csv").exists():
            enrich_season(s)

    data = pi.BookerData(seasons=range(2015, 2028))
    all_rows = []
    for target in SEASONS:
        train = [s for s in range(target - 3, target) if s in data.STINTS]
        if len(train) < 2:
            continue
        post, _ = fit_bayesian(train, target, data)
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
                "sd_off": round(r.sd_off, 2),
                "sd_def": round(r.sd_def, 2),
                "waa_off": round(k * r.impact_off * pres, 2),
                "waa_def": round(k * r.impact_def * pres, 2),
                "waa_total": round(k * r.impact_total * pres, 2),
            })
    if not all_rows:
        return
    out = pd.DataFrame(all_rows)
    out = out[out.minutes >= 250].copy()
    out["rank"] = out.groupby("season").waa_total.rank(ascending=False, method="first").astype(int)
    path = HERE / "booker_bayesian_ratings.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
