"""
Best-effort scraper for oddsportal.com NBA results moneylines.

STATUS / IMPORTANT
------------------
oddsportal.com renders its results table from an obfuscated internal API and is
protected by aggressive bot detection (Cloudflare + encoded feed responses). In
sandboxed/headless/embedded browsers the results grid does not hydrate (the page
body stays near-empty and no event rows or game links appear), so a reliable
automated scrape is not possible from this environment.

The project therefore sources historical closing moneylines from a
directly-fetchable archive in fetch_odds.py, which is the working path and
produces cache/odds_{season}.csv. This module documents the intended oddsportal
workflow for use in a real, interactive browser session where the grid hydrates,
and emits files in the identical schema so it is a drop-in replacement.

Expected per-season pages (oddsportal paginates results):
    https://www.oddsportal.com/basketball/usa/nba-{YYYY}-{YYYY+1}/results/#/page/{n}/

For each event row the grid exposes: date, home team, away team, score, and the
average home/away moneyline. Convert those to vig-removed implied probabilities
exactly as fetch_odds.ml_to_prob / normalization does, then write:
    cache/odds_{season}.csv  DATE, HOME, AWAY, ML_HOME, ML_AWAY, P_HOME, P_AWAY

To run interactively, drive a real browser (e.g. the cursor-ide-browser MCP),
collect the rendered rows per page via the accessibility snapshot or
Runtime.evaluate over the event-row DOM nodes, and pass them to rows_to_csv().
"""
from pathlib import Path

import pandas as pd

from fetch_odds import NICK_TO_ABBR, ml_to_prob, season_of  # reuse mappings

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"

SEASON_URL = ("https://www.oddsportal.com/basketball/usa/"
              "nba-{a}-{b}/results/#/page/{page}/")


def page_url(season_end_year, page=1):
    return SEASON_URL.format(a=season_end_year - 1, b=season_end_year, page=page)


def rows_to_csv(rows):
    """Write rendered oddsportal rows (list of dicts) to cache/odds_{season}.csv.

    Each row dict must have: date (YYYY-MM-DD), home, away (full team names or
    nicknames), ml_home, ml_away (American odds).
    """
    out = []
    for r in rows:
        h = _abbr(r["home"]); a = _abbr(r["away"])
        if h is None or a is None or not r.get("ml_home") or not r.get("ml_away"):
            continue
        ph, pa = ml_to_prob(r["ml_home"]), ml_to_prob(r["ml_away"])
        s = ph + pa
        out.append({
            "DATE": r["date"], "SEASON": season_of(r["date"]),
            "HOME": h, "AWAY": a, "ML_HOME": int(r["ml_home"]),
            "ML_AWAY": int(r["ml_away"]),
            "P_HOME": round(ph / s, 4), "P_AWAY": round(pa / s, 4),
        })
    df = pd.DataFrame(out)
    cols = ["DATE", "HOME", "AWAY", "ML_HOME", "ML_AWAY", "P_HOME", "P_AWAY"]
    for season, g in df.groupby("SEASON"):
        g.sort_values("DATE")[cols].to_csv(CACHE / f"odds_{season}.csv", index=False)
    return df


def _abbr(name):
    if name in NICK_TO_ABBR:
        return NICK_TO_ABBR[name]
    for nick, ab in NICK_TO_ABBR.items():           # match on trailing nickname
        if str(name).endswith(nick):
            return ab
    return None


if __name__ == "__main__":
    print(__doc__)
    print("Sample page URLs:")
    for y in (2024, 2025, 2026):
        print(" ", page_url(y))
