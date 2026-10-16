"""
Gate for BookerFormer likelihood / prior changes (audit round 2 #3). Run one config per
process (BF_LIK, BF_TAU are read at import):

    BF_LIK=legacy BF_TAU=2.0 python bf_gate.py
    BF_LIK=poss   BF_TAU=3.5 python bf_gate.py

  player : rating THROUGH s (train s-2..s) vs NEXT-season pure RAPM (s+1), Spearman, players
           >= 500 min in s+1, s = 2018..2025 (mean over seasons); also z-calibration:
           sd of (pure_{s+1} - rating_s) / posterior sd (relative only -- pure RAPM is noisy)
  team   : train s-3..s-1, held-out season s team net / wins (bookerformer_backtest.team_scores),
           s = 2020..2025
  stint  : held-out stint 90% predictive coverage (poss-weighted), same folds as team
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from forecast import player_impacts as pi
from forecast import bookerformer as bf
import bookerformer_backtest as bb


def main():
    data = pi.BookerData(seasons=range(2015, 2028))
    pure = pd.read_csv("cache/pure_rapm.csv")[["PLAYER_ID", "season", "pure_rapm"]]
    rho, zsd, mov = [], [], []
    for s in range(2018, 2026):
        post, _ = bf.fit_bookerformer([s - 2, s - 1, s], s + 1, data, mc_samples=30)
        nxt = pure[pure.season == s + 1].merge(data.PLAYERS[s + 1][["PLAYER_ID", "MINUTES"]], on="PLAYER_ID")
        nxt = nxt[nxt.MINUTES >= 500].merge(post, on="PLAYER_ID")
        rho.append(spearmanr(nxt.impact_total, nxt.pure_rapm)[0])
        z = (nxt.pure_rapm - nxt.impact_total) / np.hypot(nxt.sd_off, nxt.sd_def)
        zsd.append(float(z.std()))
        mov.append(float(post.impact_total.std()))
    nr, wr, r2, cov = [], [], [], []
    for s in range(2020, 2026):
        train = [s - 3, s - 2, s - 1]
        post, model = bf.fit_bookerformer(train, s, data, mc_samples=30)
        a, b, _, c = bb.team_scores(data, dict(zip(post.PLAYER_ID, post.impact_total)), s, train)
        nr.append(a); wr.append(b); r2.append(c)
        d = data.STINTS[s]
        mu, sd, keep = bf.predict_offense_points(model, model._fd, list(d.home), list(d.away), mc_samples=30,
                                                 poss=d.POSS.to_numpy())
        act, w = d.Y_OFF_HOME.to_numpy()[keep], d.POSS.to_numpy()[keep]
        cov.append(float(np.average((np.abs(act - mu) <= 1.645 * sd).astype(float), weights=w)))
    print(f"BF_LIK={bf.BF_LIK} TAU={bf.TAU_RATING}: player rho(next pure) {np.mean(rho):.4f} "
          f"[{', '.join(f'{x:.3f}' for x in rho)}] | z-sd {np.mean(zsd):.2f} | rating sd {np.mean(mov):.2f} || "
          f"team net RMSE {np.mean(nr):.3f} wins RMSE {np.mean(wr):.2f} R2 {np.mean(r2):.3f} | stint cov90 {np.mean(cov):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
