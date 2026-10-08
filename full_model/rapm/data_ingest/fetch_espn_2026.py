"""Fetch full-season 2025-26 player totals from ESPN (stats.nba.com is IP-blocked here;
ESPN is reachable). 12 paginated calls -> per-player season counts + MINUTES + steals/
blocks (which the CDN parquet lacked). Map ESPN names -> BOOKER PLAYER_IDs and write
cache/espn_pbp_2026.csv in the pbp_skills counts schema."""
import sys, time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

RAPM = Path("/Users/reececalvin/BOOKER/BOOKER/full_model/rapm")
sys.path.insert(0, str(RAPM))
from forecast import player_impacts as pi
CACHE = RAPM / "cache"
H = {"User-Agent": "Mozilla/5.0"}
URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/statistics/byathlete"

# positional indices into each category's `totals` (decoded from the label lists)
GEN_MIN, GEN_REB, GEN_PF = 8, 9, 10
OFF_FGA, OFF_3PM, OFF_FTM, OFF_FTA, OFF_AST, OFF_TO = 14, 15, 17, 18, 19, 20
DEF_STL, DEF_BLK = 2, 3


def fetch():
    rows = []
    for page in range(1, 13):
        p = {"region": "us", "lang": "en", "contentorigin": "espn", "isqualified": "false",
             "page": page, "limit": 50, "season": 2026, "seasontype": 2}
        j = requests.get(URL, params=p, headers=H, timeout=30).json()
        for a in j.get("athletes", []):
            cats = {c["name"]: c.get("totals", []) for c in a.get("categories", [])}
            g, o, d = cats.get("general", []), cats.get("offensive", []), cats.get("defensive", [])
            if len(g) <= GEN_PF or len(o) <= OFF_TO or len(d) <= DEF_BLK:
                continue
            def fv(arr, i):
                try: return float(str(arr[i]).replace(",", ""))
                except Exception: return 0.0
            rows.append({"name": a["athlete"]["displayName"],
                         "minutes": fv(g, GEN_MIN), "reb": fv(g, GEN_REB), "pf": fv(g, GEN_PF),
                         "fga": fv(o, OFF_FGA), "fg3m": fv(o, OFF_3PM), "ftm": fv(o, OFF_FTM),
                         "fta": fv(o, OFF_FTA), "ast": fv(o, OFF_AST), "tov": fv(o, OFF_TO),
                         "stl": fv(d, DEF_STL), "blk": fv(d, DEF_BLK)})
        time.sleep(0.4)
    return pd.DataFrame(rows)


def main():
    e = fetch()
    print(f"ESPN 2026: {len(e)} players; total minutes {int(e.minutes.sum())} (full season ~590k)")
    # name -> PLAYER_ID from BOOKER ratings (NBA ids)
    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")[["PLAYER_ID", "player"]].drop_duplicates("PLAYER_ID")
    nm2id = {pi.norm_name(p): int(i) for i, p in zip(rat.PLAYER_ID, rat.player)}
    e["PLAYER_ID"] = e.name.map(lambda n: nm2id.get(pi.norm_name(n)))
    matched = e.PLAYER_ID.notna().sum()
    print(f"mapped to PLAYER_ID: {matched}/{len(e)}")
    print("unmatched (sample):", e[e.PLAYER_ID.isna()].name.head(12).tolist())
    e = e.dropna(subset=["PLAYER_ID"]).copy()
    e["PLAYER_ID"] = e.PLAYER_ID.astype(int)
    e["season"] = 2026
    # create36 needs assisted-shot value (3 vs 2); box totals lack the split -> approximate
    # each assist at ~2.3 pts (league mix ~40% assisted-3s), ast3 ~ 0.4*ast.
    e["create"] = (e.ast * 2.3).round(1)
    e["ast3"] = (e.ast * 0.4).round(0)
    out = e[["PLAYER_ID", "season", "ast", "ast3", "create", "stl", "blk", "reb",
             "fga", "fta", "ftm", "tov", "pf", "minutes"]]
    out.to_csv(CACHE / "espn_pbp_2026.csv", index=False)
    print(f"wrote cache/espn_pbp_2026.csv ({len(out)} players)")
    # spot check
    for nm in ["Luka", "Jokic", "Gilgeous", "Jaren Jackson"]:
        r = out.merge(e[["PLAYER_ID", "name"]], on="PLAYER_ID")
        r = r[r.name.str.contains(nm, case=False, na=False)]
        if len(r):
            x = r.iloc[0]
            print(f"  {x['name']}: min {x.minutes:.0f} ast {x.ast:.0f} stl {x.stl:.0f} blk {x.blk:.0f} "
                  f"reb {x.reb:.0f} tov {x.tov:.0f} ft {x.ftm:.0f}/{x.fta:.0f}")


if __name__ == "__main__":
    main()
