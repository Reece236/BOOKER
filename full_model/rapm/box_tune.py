"""
Tune / validate the sequential filter's box-score channel (BOX_TAU, BOX_KAPPA).

  python box_tune.py tune      # grid on 2019-21 (tuning seasons)
  python box_tune.py test T K  # stint-only vs (T, K) on 2022-26 (test seasons)

Metrics per config: midseason & late-season Spearman of the filter's o+d (pre-game state at
the cutoff) vs rest-of-season pure RAPM (players >= 500 later minutes, prior-rated), and
game log-loss of a pooled 2-parameter logistic on the strict / pregame margins.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from forecast import player_impacts as pi
from forecast import minutes_share as ms
from forecast import sequential as sq
import compare_darko as cd


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run(seasons, configs):
    data = pi.BookerData()
    rat = pd.read_csv(sq.RAPM / "booker_bookerformer_ratings.csv")
    allp = ms.load()
    panel = sq.availability_panel()
    ros = {}
    res = []
    for name, tau, kap in configs:
        G_all, rho = [], {"mid": [], "late": []}
        for s in seasons:
            proj = allp[allp.season == s]
            box = sq.box_inputs(s, rat) if tau is not None else None
            G, Pf, _ = sq.run_season(data, s, rat, proj, float(proj.repl.iloc[0]), verbose=False,
                                     avail_model=sq.fit_availability(panel, s), box=box,
                                     box_tau=tau, box_kappa=kap)
            gm = data.GAMES[s]; gm = gm[gm.SEASON_TYPE == "Regular Season"]
            G = G.merge(gm[["GAME_ID", "HOME_WIN"]], on="GAME_ID")
            G_all.append(G)
            dates = sorted(gm.DATE.unique())
            Pf["v"] = Pf.o + Pf.d
            r1 = pi.ratings_through(s - 1, rat).set_index("PLAYER_ID").impact_total
            for lab, cut in (("mid", dates[len(dates) // 2]), ("late", dates[int(len(dates) * 0.75)])):
                if (s, lab) not in ros:
                    o = cd.rest_of_season(data, s, cut)
                    ros[(s, lab)] = o[(o["min"] >= 500) & o.PLAYER_ID.isin(r1.index)]
                o = ros[(s, lab)].copy()
                o["est"] = o.PLAYER_ID.map(Pf[Pf.date >= cut].sort_values("date").groupby("PLAYER_ID").v.first())
                o = o.dropna(subset=["est"])
                rho[lab].append((spearmanr(o.est, o.pure)[0], len(o)))
        G = pd.concat(G_all)
        out = {"config": name}
        from sklearn.linear_model import LogisticRegression
        for mode in ("strict", "pregame"):
            x = (G[f"net_home_{mode}"] - G[f"net_away_{mode}"]).values.reshape(-1, 1)
            lr = LogisticRegression(C=1e6).fit(x, G.HOME_WIN)
            out[f"LL_{mode}"] = round(ll(lr.predict_proba(x)[:, 1], G.HOME_WIN.values), 4)
        for lab in ("mid", "late"):
            out[f"rho_{lab}"] = round(float(np.average([r for r, _ in rho[lab]], weights=[n for _, n in rho[lab]])), 4)
        print(out, flush=True)
        res.append(out)
    return pd.DataFrame(res)


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        cfg = [("stint-only", None, None)] + [(f"tau{t}_k{k}", t, k) for t in (3.0, 4.5) for k in (0.5, 1.0)]
        print(run(range(2019, 2022), cfg).to_string(index=False))
    else:
        t, k = float(sys.argv[2]), float(sys.argv[3])
        print(run(range(2022, 2027), [("stint-only", None, None), (f"tau{t}_k{k}", t, k)]).to_string(index=False))
