"""
Project 2026-27 minutes for players_2027.csv (roster/team assignments untouched).

The naive build carried each player's LATEST-season minutes forward, which (a) used the
partial 2026 stint snapshot (~64% of the season -- Wemby 1308 vs true 1866) and (b) has
no injury-recovery logic (Tatum's 342 injured minutes became his 2027 projection).

Rule (OOS-validated on 3,130 player-season transitions, 2019-2025: RMSE 547 -> 518 vs
naive overall; 583 -> 502 in the injury-recovery class):
  history h = [2026 FULL-season minutes (ESPN), 2025, 2024]
  wmean    = recency-weighted mean over played seasons (.55/.30/.15 renormalized)
  proj     = max(wmean, 0.50 * max(h))       # recovery toward the established norm
  age      <=23 x1.10, >=36 x0.82, >=33 x0.90; clip [200, 3100]
Players with no NBA history (rookies) keep their existing default.

Run: python data_ingest/project_minutes_2027.py [--write]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
sys.path.insert(0, str(RAPM))

RECOV = 0.50
W = np.array([0.55, 0.30, 0.15])


def main(write=False):
    from forecast import uncertainty as unc
    p27 = pd.read_csv(CACHE / "players_2027.csv")
    espn = pd.read_csv(CACHE / "espn_pbp_2026.csv")
    m26 = dict(zip(espn.PLAYER_ID.astype(int), espn.minutes.astype(float)))
    m25 = dict(zip(*[pd.read_csv(CACHE / "players_2025.csv")[c] for c in ("PLAYER_ID", "MINUTES")]))
    m24 = dict(zip(*[pd.read_csv(CACHE / "players_2024.csv")[c] for c in ("PLAYER_ID", "MINUTES")]))

    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    a = unc.attach_attrs(rat.drop_duplicates("PLAYER_ID", keep="last"))
    age_ref = {int(p): (int(s), float(ag)) for p, s, ag in zip(a.PLAYER_ID, a.season, a.age)
               if pd.notna(ag)}

    out, changed = [], []
    for r in p27.itertuples():
        pid = int(r.PLAYER_ID)
        h = np.array([m26.get(pid) or 0.0, m25.get(pid) or 0.0, m24.get(pid) or 0.0])
        if h.max() <= 0:
            out.append(float(r.MINUTES))           # rookie/no history: keep default
            continue
        mask = h > 0
        wm = float(np.dot(W[mask], h[mask]) / W[mask].sum())
        proj = max(wm, RECOV * float(h.max()))
        ar = age_ref.get(pid)
        if ar is not None:
            age = ar[1] + (2027 - ar[0])
            if age <= 23: proj *= 1.10
            elif age >= 36: proj *= 0.82
            elif age >= 33: proj *= 0.90
        proj = float(np.clip(proj, 200, 3100))
        out.append(round(proj, 1))
        if abs(proj - float(r.MINUTES)) > 400:
            changed.append((r.NAME, float(r.MINUTES), proj))
    p27["MINUTES"] = out
    print(f"{len(p27)} players; {len(changed)} moved by >400 min. Biggest moves:")
    for nm, old, new in sorted(changed, key=lambda t: -abs(t[2] - t[1]))[:12]:
        print(f"  {nm:26s} {old:7.0f} -> {new:7.0f}")
    if write:
        p27.to_csv(CACHE / "players_2027.csv", index=False)
        print("WROTE cache/players_2027.csv")


if __name__ == "__main__":
    main(write="--write" in sys.argv)
