"""Per-player PER-GAME ratings keyed by GAME_ID -> cache/player_game_ratings.csv.

For each stint, the leave-one-out residual credits player p with what his lineups
did beyond what the other nine players + home court predict:
    home p:  g = impact_p + (Y - E[Y]);  away p: g = impact_p - (Y - E[Y])
    E[Y] = HCA + sum(home impacts) - sum(away impacts)      (additive convention)
Per game: possession-weighted mean over the player's stints; then each
player-season is anchored (additive shift) so his poss-weighted season mean
equals his season impact_total -- game ratings orbit the season rating.

Output columns: season, GAME_ID, PLAYER_ID, poss, g  (g = per-100 impact that game)
"""
from __future__ import annotations
from pathlib import Path


import pandas as pd

from . import player_impacts as pi

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
OUT = CACHE / "player_game_ratings.csv"
MIN_POSS_GAME = 10


def build(seasons=tuple(range(2018, 2027))):
    bk = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    frames = []
    for s in seasons:
        sp = CACHE / f"stints_{s}.csv"
        if not sp.exists():
            continue
        st = pd.read_csv(sp)
        # playoff stints live in a separate file for 2018-2025 (the historical
        # feed was regular-season only; build_playoff_stints.py backfills them)
        pop = CACHE / f"stints_po_{s}.csv"
        if pop.exists():
            po = pd.read_csv(pop)
            po = po[~po.GAME_ID.isin(set(st.GAME_ID))]
            st = pd.concat([st, po], ignore_index=True)
        imp = dict(zip(bk[bk.season == s].PLAYER_ID, bk[bk.season == s].impact_total))
        rows = {}
        for r in st.itertuples():
            try:
                hl = [int(x) for x in str(r.HOME_LINEUP).split(",")]
                al = [int(x) for x in str(r.AWAY_LINEUP).split(",")]
            except ValueError:
                continue
            if len(hl) != 5 or len(al) != 5 or not (r.POSS == r.POSS) or r.POSS <= 0:
                continue
            ey = pi.HOME_COURT_ADV + sum(imp.get(p, 0.0) for p in hl) \
                - sum(imp.get(p, 0.0) for p in al)
            resid = float(r.Y) - ey
            for p in hl:
                k = (int(r.GAME_ID), p)
                w, g = rows.get(k, (0.0, 0.0))
                rows[k] = (w + r.POSS, g + r.POSS * (imp.get(p, 0.0) + resid))
            for p in al:
                k = (int(r.GAME_ID), p)
                w, g = rows.get(k, (0.0, 0.0))
                rows[k] = (w + r.POSS, g + r.POSS * (imp.get(p, 0.0) - resid))
        d = pd.DataFrame([{"season": s, "GAME_ID": gid, "PLAYER_ID": p,
                           "poss": w, "g": v / w}
                          for (gid, p), (w, v) in rows.items() if w >= MIN_POSS_GAME])
        if d.empty:
            print(f"  {s}: no usable stints, skipped")
            continue
        # single-game LOO residuals are wildly noisy (raw p10/p90 ~ +-38): shrink
        # each game toward the player's season rating by possessions, same logic
        # as the trajectory form dots. k = poss/(poss+P0), P0=150 -> a 70-poss
        # starter game keeps ~32% of its raw deviation.
        P0 = 150.0
        base = d.PLAYER_ID.map(imp).fillna(0.0)
        k = d.poss / (d.poss + P0)
        d["g"] = base + k * (d.g - base)
        # anchor: player-season poss-weighted mean of REGULAR-SEASON games ==
        # season impact_total. Playoff games take the same shift but are NOT
        # part of the anchor -- their deviation from the season rating is the
        # playoff-rise/fall signal we want to measure, not calibrate away.
        d["_reg"] = ~d.GAME_ID.astype(str).str.startswith("4")
        d["_wg"] = d.g * d.poss
        reg = d[d._reg]
        agg = reg.groupby("PLAYER_ID").agg(wg=("_wg", "sum"), w=("poss", "sum"))
        shift = {pid: imp[pid] - r.wg / r.w for pid, r in agg.iterrows()
                 if pid in imp and r.w > 0}
        d["g"] = d.g + d.PLAYER_ID.map(shift).fillna(0.0)
        d = d.drop(columns=["_wg", "_reg"])
        d["g"] = d.g.clip(-45, 45).round(1)
        frames.append(d)
        print(f"  {s}: {len(d)} player-games, {d.GAME_ID.nunique()} games")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    build()
