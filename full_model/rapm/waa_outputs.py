"""
BOOKER WAA -- final outputs.

1. Headline player Wins Above Average ratings for the latest season (2024-25),
   from the box-prior-blended RAPM that the backtest validated.
2. A diagnostics PDF: out-of-sample win/net-rating calibration, per-fold skill,
   model-vs-benchmark comparison, and the player WAA leaderboard.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ROOT = HERE.parent.parent
PLAYER_DATA = ROOT / "full_model" / "nba_player_data_2015-2025.csv"
TEAM_PRED = ROOT / "full_model" / "team_predictions.csv"

SEASON = 2025
SEC_PER_POSS = 28.8
PRIOR_BASE, PRIOR_K, PRIOR_CLIP = -1.0, 150.0, (-12.0, 14.0)
ALPHA = 2000

COL_MAIN, COL_ACC, COL_GOOD, COL_BAD = "#2E86AB", "#E8451E", "#2A9D8F", "#E63946"


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


def parse(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def build_player_ratings():
    stints = pd.read_csv(CACHE / f"stints_{SEASON}.csv")
    players = pd.read_csv(CACHE / f"players_{SEASON}.csv")
    teams = pd.read_csv(CACHE / f"teams_{SEASON}.csv")

    bk = pd.read_csv(PLAYER_DATA)
    bk = bk[bk.season == SEASON].dropna(subset=["box"]).copy()
    bk["nm"] = bk.playerName.map(norm_name)
    box = {nm: np.average(g.box, weights=g.minutesPlayed.clip(lower=1))
           for nm, g in bk.groupby("nm")}

    players["nm"] = players.NAME.map(norm_name)
    players["raw_bpm"] = players.nm.map(box)
    n_match = players.raw_bpm.notna().sum()
    players["raw_bpm"] = players.raw_bpm.clip(*PRIOR_CLIP).fillna(PRIOR_BASE)
    sh = players.MINUTES / (players.MINUTES + PRIOR_K)
    players["prior"] = players.raw_bpm * sh + PRIOR_BASE * (1 - sh)
    prior_by_id = dict(zip(players.PLAYER_ID, players.prior))

    home = stints.HOME_LINEUP.map(parse)
    away = stints.AWAY_LINEUP.map(parse)
    all_ids = sorted({p for lu in home for p in lu} | {p for lu in away for p in lu})
    col = {p: i for i, p in enumerate(all_ids)}
    rows, cols, vals = [], [], []
    for i, (hl, al) in enumerate(zip(home, away)):
        for p in hl:
            rows.append(i); cols.append(col[p]); vals.append(1.0)
        for p in al:
            rows.append(i); cols.append(col[p]); vals.append(-1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(len(stints), len(all_ids)))
    y = stints.Y.values; w = stints.POSS.values
    b0 = np.array([prior_by_id.get(p, PRIOR_BASE) for p in all_ids])
    ridge = Ridge(alpha=ALPHA, fit_intercept=True)
    ridge.fit(X, y - X.dot(b0), sample_weight=w)
    blended = b0 + ridge.coef_
    impact = dict(zip(all_ids, blended))

    # net->wins slope k (actual)
    tp = pd.read_csv(TEAM_PRED)
    tp = tp[tp.season == SEASON]
    wins_by_abbr = dict(zip(tp.team_abbr, tp.actual_wins))
    act = teams.set_index("TEAM_ID")
    tn, tw = [], []
    for tid, ab in zip(teams.TEAM_ID, teams.ABBR):
        if ab in wins_by_abbr:
            tn.append(act.loc[tid, "ACTUAL_NET"]); tw.append(wins_by_abbr[ab])
    k, c = np.polyfit(tn, tw, 1)

    pl = players.copy()
    pl["impact_per100"] = pl.PLAYER_ID.map(impact)
    tmin = pl.groupby("TEAM_ID").MINUTES.transform("sum")
    pl["presence"] = pl.MINUTES / (tmin / 5.0)
    pl["waa_wins"] = k * pl.impact_per100 * pl.presence       # wins above .500 contributed
    pl["team"] = pl.TEAM_ID.map(dict(zip(teams.TEAM_ID, teams.ABBR)))
    out = pl[["NAME", "team", "MINUTES", "prior", "impact_per100", "waa_wins"]].copy()
    out.columns = ["player", "team", "minutes", "prior_bpm", "waa_per100", "waa_wins"]
    out = out.sort_values("waa_per100", ascending=False).round(
        {"minutes": 0, "prior_bpm": 2, "waa_per100": 2, "waa_wins": 2})
    out.to_csv(HERE / "booker_waa_player_ratings.csv", index=False)
    print(f"player ratings: {len(out)} players, {n_match} matched BPM, k={k:.2f} wins/net")
    return out


def diagnostics(ratings):
    bt = pd.read_csv(HERE / "waa_backtest_team_predictions.csv")
    fm = pd.read_csv(HERE / "waa_backtest_metrics.csv")
    with PdfPages(HERE / "diagnostics_waa.pdf") as pdf:
        # page 1: out-of-sample wins calibration
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        ax[0].scatter(bt.actual_wins, bt.pred_wins, c=COL_MAIN, alpha=0.6, edgecolor="w")
        lim = [10, 75]
        ax[0].plot(lim, lim, "--", color=COL_ACC, lw=2)
        rmse = np.sqrt(np.mean((bt.pred_wins - bt.actual_wins) ** 2))
        r2 = np.corrcoef(bt.actual_wins, bt.pred_wins)[0, 1] ** 2
        ax[0].set(xlabel="Actual wins", ylabel="WAA predicted wins (out-of-sample)",
                  xlim=lim, ylim=lim,
                  title=f"Out-of-sample win forecast 2018-2025\nRMSE={rmse:.1f}  R2={r2:.2f}")
        ax[0].grid(alpha=.3)
        ax[1].scatter(bt.actual_net, bt.pred_net, c=COL_GOOD, alpha=0.6, edgecolor="w")
        nl = [-15, 15]
        ax[1].plot(nl, nl, "--", color=COL_ACC, lw=2)
        ax[1].set(xlabel="Actual net rating", ylabel="WAA predicted net rating",
                  xlim=nl, ylim=nl, title="Out-of-sample team net rating")
        ax[1].grid(alpha=.3)
        pdf.savefig(fig); plt.close(fig)

        # page 2: per-fold skill + benchmark
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        folds = fm[fm.season != "POOLED"].copy()
        folds["season"] = folds.season.astype(int)
        ax[0].bar(folds.season.astype(str), folds.wins_rmse, color=COL_MAIN)
        ax[0].axhline(11.99, ls="--", color=COL_BAD, label="predict-41 baseline")
        ax[0].axhline(9.01, ls=":", color="gray", label="old box model (retrodictive)")
        ax[0].set(ylabel="Wins RMSE", title="Per-season out-of-sample wins RMSE")
        ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
        ax[1].bar(folds.season.astype(str), folds.wins_r2, color=COL_GOOD)
        ax[1].set(ylabel="Wins R2", title="Per-season out-of-sample wins R2", ylim=[0, 1])
        ax[1].grid(alpha=.3, axis="y")
        pdf.savefig(fig); plt.close(fig)

        # page 3: top-30 player WAA
        top = ratings.head(30).iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(top.player + " (" + top.team + ")", top.waa_per100, color=COL_GOOD)
        ax.set(xlabel="Blended RAPM impact (pts / 100 poss above average)",
               title=f"Top 30 players by on-court WAA impact, {SEASON-1}-{str(SEASON)[2:]}")
        ax.grid(alpha=.3, axis="x")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("wrote diagnostics_waa.pdf")


if __name__ == "__main__":
    r = build_player_ratings()
    diagnostics(r)
    print("\nTop 15 players by WAA impact:")
    print(r.head(15).to_string(index=False))
