"""
On-ball / off-ball GRAVITY proxies (leakage-free).

True gravity needs player-tracking data (defender attention/positioning), which we
don't have. We approximate it from a player's OWN shot diet + scoring -- what actually
makes a defense respect (and collapse toward) them:

  off_gravity = catch-and-shoot 3 POINTS per 36 -- spot-up shooting threat that keeps
                defenders attached off the ball (volume x accuracy). Curry/Klay-types.
  on_gravity  = self-created + driving POINTS per 36 -- on-ball attack threat that
                bends/collapses the defense (pull-ups, step-backs, drives). Luka/SGA-types.

Both are per-36 RATES (minutes-independent). Low-minute reliability is handled by the
grade's Bayesian shrinkage downstream. These are PROXIES, not measured defender draw.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import shot_quality as sq

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
OUT = CACHE / "gravity.csv"
ONBALL = {"pullup", "stepback", "fadeaway", "driving", "floating"}   # ball-handler-created


def build(seasons=range(2018, 2027), min_shots=30, min_minutes=200):
    rat = pd.read_csv(HERE.parent / "booker_bookerformer_ratings.csv")
    mins = {(int(r.season), int(r.PLAYER_ID)): float(r.minutes) for r in rat.itertuples()}
    rows = []
    for s in seasons:
        p = CACHE / f"shotdetail_{s}.csv"
        if not p.exists():
            continue
        df = sq._featurize(pd.read_csv(p, low_memory=False))
        df["ptval"] = df.is3 * 3 + (1 - df.is3) * 2
        df["cs"] = (df.bucket == "jump").astype(int)                      # catch-and-shoot / spot-up
        df["cs3"] = ((df.is3 == 1) & (df.bucket == "jump")).astype(int)   # catch-and-shoot 3
        df["onb"] = df.bucket.isin(ONBALL).astype(int)
        df["cs3_pts"] = df.made * df.cs3 * 3
        df["csAll_pts"] = df.made * df.cs * df.ptval                       # C&S points incl spot-up 2s
        df["cs_att"] = df.cs
        df["cs_made"] = df.made * df.cs
        df["onb_pts"] = df.made * df.ptval * df.onb
        g = df.groupby("PLAYER_ID").agg(player=("PLAYER_NAME", "first"), shots=("made", "size"),
                                        cs3_pts=("cs3_pts", "sum"), csAll_pts=("csAll_pts", "sum"),
                                        cs_att=("cs_att", "sum"), cs_made=("cs_made", "sum"),
                                        onb_pts=("onb_pts", "sum"))
        lg = float(g.cs_made.sum() / max(g.cs_att.sum(), 1.0))            # league C&S make%
        for pid, r in g.iterrows():
            m = mins.get((int(s), int(pid)), 0.0)
            if m < min_minutes or r.shots < min_shots:
                continue
            u = m / 36.0
            att = float(r.cs_att)
            # spacing THREAT = defender respect: EB-shrunk C&S make% centered on league
            # (non-shooters go NEGATIVE = shrink the floor), gated by attempt volume.
            acc = (float(r.cs_made) + 60.0 * lg) / (att + 60.0)
            threat = (acc - lg) * min(att / 100.0, 1.0)
            rows.append({"season": int(s), "PLAYER_ID": int(pid),
                         "off_gravity": round(float(r.cs3_pts) / u, 2),
                         "on_gravity": round(float(r.onb_pts) / u, 2),
                         "csAll_gravity": round(float(r.csAll_pts) / u, 2),
                         "space_threat": round(threat, 4)})
        print(f"  gravity {s}: {sum(1 for x in rows if x['season'] == s)} players")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} player-seasons)")
    return out


if __name__ == "__main__":
    build()
