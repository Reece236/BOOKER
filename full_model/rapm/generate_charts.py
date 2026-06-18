"""
BOOKER WAA -- charts.

A) Win-prediction performance by year (out-of-sample backtest).
B) Player WAA ranks by year (per-season box-prior-blended RAPM).

Writes per-season ratings to booker_waa_ratings_by_year.csv and two multi-page
PDFs plus standalone PNGs of the headline figures.
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
from matplotlib.cm import get_cmap
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ROOT = HERE.parent.parent
PLAYER_DATA = ROOT / "full_model" / "nba_player_data_2015-2025.csv"
TEAM_PRED = ROOT / "full_model" / "team_predictions.csv"

SEASONS = list(range(2015, 2027))
BOX_SEASONS = set(range(2015, 2026))   # seasons with same-season box priors
SEC_PER_POSS = 28.8
PRIOR_BASE, PRIOR_K, PRIOR_CLIP, ALPHA = -1.0, 150.0, (-12.0, 14.0), 2000

C_MAIN, C_ACC, C_GOOD, C_BAD, C_GREY = "#2E86AB", "#E8451E", "#2A9D8F", "#E63946", "#9AA0A6"
BASE_RMSE, OLD_RMSE = 11.99, 9.01   # predict-41 baseline, old box model (retrodictive)


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


# ---------------------------------------------------------------------------
# A) per-season player ratings
# ---------------------------------------------------------------------------
def season_rating(season, box_by, box_fallback=None):
    stints = pd.read_csv(CACHE / f"stints_{season}.csv")
    players = pd.read_csv(CACHE / f"players_{season}.csv")
    teams = pd.read_csv(CACHE / f"teams_{season}.csv")

    players["nm"] = players.NAME.map(norm_name)
    # seasons without same-season box priors (2025-26) carry forward each
    # player's most recent prior-season box score.
    fb = box_fallback or {}
    players["raw_bpm"] = players.nm.map(
        lambda k: box_by.get((k, season), fb.get(k)))
    players["raw_bpm"] = players.raw_bpm.clip(*PRIOR_CLIP).fillna(PRIOR_BASE)
    sh = players.MINUTES / (players.MINUTES + PRIOR_K)
    players["prior"] = players.raw_bpm * sh + PRIOR_BASE * (1 - sh)
    prior_by_id = dict(zip(players.PLAYER_ID, players.prior))

    home = stints.HOME_LINEUP.map(parse); away = stints.AWAY_LINEUP.map(parse)
    all_ids = sorted({p for lu in home for p in lu} | {p for lu in away for p in lu})
    col = {p: i for i, p in enumerate(all_ids)}
    rows, cols, vals = [], [], []
    for i, (hl, al) in enumerate(zip(home, away)):
        for p in hl:
            rows.append(i); cols.append(col[p]); vals.append(1.0)
        for p in al:
            rows.append(i); cols.append(col[p]); vals.append(-1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(len(stints), len(all_ids)))
    b0 = np.array([prior_by_id.get(p, PRIOR_BASE) for p in all_ids])
    ridge = Ridge(alpha=ALPHA, fit_intercept=True)
    ridge.fit(X, stints.Y.values - X.dot(b0), sample_weight=stints.POSS.values)
    impact = dict(zip(all_ids, b0 + ridge.coef_))

    tp = pd.read_csv(TEAM_PRED); tp = tp[tp.season == season]
    wins_by_abbr = dict(zip(tp.team_abbr, tp.actual_wins))
    act = teams.set_index("TEAM_ID")
    tn, tw = [], []
    for tid, ab in zip(teams.TEAM_ID, teams.ABBR):
        if ab in wins_by_abbr:
            tn.append(act.loc[tid, "ACTUAL_NET"]); tw.append(wins_by_abbr[ab])
    k = np.polyfit(tn, tw, 1)[0] if len(tn) > 2 else 2.3

    pl = players.copy()
    pl["impact_per100"] = pl.PLAYER_ID.map(impact)
    tmin = pl.groupby("TEAM_ID").MINUTES.transform("sum")
    pl["presence"] = pl.MINUTES / (tmin / 5.0)
    pl["waa_wins"] = k * pl.impact_per100 * pl.presence
    pl["team"] = pl.TEAM_ID.map(dict(zip(teams.TEAM_ID, teams.ABBR)))
    pl["season"] = season
    pl = pl[pl.MINUTES >= 250].copy()          # rotation players only for ranks
    pl["rank"] = pl.waa_wins.rank(ascending=False, method="first").astype(int)
    return pl[["season", "rank", "PLAYER_ID", "NAME", "team", "MINUTES",
               "prior", "impact_per100", "waa_wins"]]


def build_all_ratings():
    bk = pd.read_csv(PLAYER_DATA).dropna(subset=["box"]).copy()
    bk["nm"] = bk.playerName.map(norm_name)
    box_by = {(nm, ss): np.average(g.box, weights=g.minutesPlayed.clip(lower=1))
              for (nm, ss), g in bk.groupby(["nm", "season"])}
    latest_box = {}
    for (nm, ss) in sorted(box_by, key=lambda kv: kv[1]):
        latest_box[nm] = box_by[(nm, ss)]      # highest season wins
    parts = [season_rating(s, box_by,
                           box_fallback=(latest_box if s not in BOX_SEASONS else None))
             for s in SEASONS if (CACHE / f"stints_{s}.csv").exists()]
    allr = pd.concat(parts, ignore_index=True)
    allr_out = allr.rename(columns={"NAME": "player", "MINUTES": "minutes"}).round(
        {"minutes": 0, "prior": 2, "impact_per100": 2, "waa_wins": 2})
    allr_out.to_csv(HERE / "booker_waa_ratings_by_year.csv", index=False)
    print(f"ratings by year: {len(allr_out)} rows across {len(SEASONS)} seasons")
    return allr


# ---------------------------------------------------------------------------
# B) win-prediction performance charts
# ---------------------------------------------------------------------------
def chart_win_performance():
    bt = pd.read_csv(HERE / "waa_backtest_team_predictions.csv")
    fm = pd.read_csv(HERE / "waa_backtest_metrics.csv")
    folds = fm[fm.season != "POOLED"].copy(); folds["season"] = folds.season.astype(int)
    seasons = sorted(bt.season.unique())
    cmap = get_cmap("viridis")

    with PdfPages(HERE / "charts_win_prediction.pdf") as pdf:
        # ---- page 1: predicted vs actual, small multiples per season ----------
        n = len(seasons); ncol = 4; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.6 * nrow))
        axes = np.array(axes).reshape(-1)
        for ax, yr in zip(axes, seasons):
            d = bt[bt.season == yr]
            rmse = np.sqrt(np.mean((d.pred_wins - d.actual_wins) ** 2))
            r2 = np.corrcoef(d.actual_wins, d.pred_wins)[0, 1] ** 2
            ax.scatter(d.actual_wins, d.pred_wins, color=C_MAIN, alpha=.8, edgecolor="w", s=45)
            ax.plot([10, 75], [10, 75], "--", color=C_ACC, lw=1.5)
            ax.set(xlim=[10, 75], ylim=[10, 75],
                   title=f"{yr-1}-{str(yr)[2:]}   RMSE={rmse:.1f}  R\u00b2={r2:.2f}")
            ax.grid(alpha=.25)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Out-of-sample win forecast: predicted vs actual, by season",
                     fontsize=15, y=1.0)
        fig.supxlabel("Actual wins"); fig.supylabel("WAA predicted wins")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- page 2: RMSE by year vs benchmarks -------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(folds))
        ax.bar(x, folds.wins_rmse, color=C_MAIN, label="BOOKER WAA (out-of-sample)")
        ax.axhline(BASE_RMSE, ls="--", color=C_BAD, lw=2, label=f"predict-41 baseline ({BASE_RMSE})")
        ax.axhline(OLD_RMSE, ls=":", color=C_GREY, lw=2, label=f"old box model, retrodictive ({OLD_RMSE})")
        for xi, v in zip(x, folds.wins_rmse):
            ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([f"{s-1}-{str(s)[2:]}" for s in folds.season])
        ax.set(ylabel="Wins RMSE", title="Win-prediction error by season (lower = better)")
        ax.legend(); ax.grid(alpha=.25, axis="y")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- page 3: R2 and calibration slope by year -------------------------
        fig, ax1 = plt.subplots(figsize=(12, 6))
        x = np.arange(len(folds)); wbar = .4
        ax1.bar(x - wbar/2, folds.wins_r2, wbar, color=C_GOOD, label="Wins R\u00b2")
        ax1.set(ylabel="Wins R\u00b2", ylim=[0, 1]); ax1.set_ylabel("Wins R\u00b2", color=C_GOOD)
        ax2 = ax1.twinx()
        ax2.bar(x + wbar/2, folds.wins_slope, wbar, color=C_ACC, label="Calibration slope")
        ax2.axhline(1.0, ls="--", color=C_GREY); ax2.set_ylabel("Calibration slope (1=ideal)", color=C_ACC)
        ax2.set_ylim([0, 1.4])
        ax1.set_xticks(x); ax1.set_xticklabels([f"{s-1}-{str(s)[2:]}" for s in folds.season])
        ax1.set_title("Out-of-sample R\u00b2 and calibration slope by season")
        ax1.grid(alpha=.25, axis="y")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- page 4: pooled scatter colored by season -------------------------
        fig, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(bt.actual_wins, bt.pred_wins, c=bt.season, cmap="viridis",
                        alpha=.8, edgecolor="w", s=55)
        ax.plot([12, 74], [12, 74], "--", color=C_ACC, lw=2)
        rmse = np.sqrt(np.mean((bt.pred_wins - bt.actual_wins) ** 2))
        r2 = np.corrcoef(bt.actual_wins, bt.pred_wins)[0, 1] ** 2
        ax.set(xlim=[12, 74], ylim=[12, 74], xlabel="Actual wins",
               ylabel="WAA predicted wins (out-of-sample)",
               title=f"Pooled 2018-2025  (239 team-seasons)\nRMSE={rmse:.1f}  R\u00b2={r2:.2f}")
        ax.grid(alpha=.25); plt.colorbar(sc, label="season")
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # standalone PNGs
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(folds))
    ax.bar(x, folds.wins_rmse, color=C_MAIN, label="BOOKER WAA (out-of-sample)")
    ax.axhline(BASE_RMSE, ls="--", color=C_BAD, lw=2, label=f"predict-41 ({BASE_RMSE})")
    ax.axhline(OLD_RMSE, ls=":", color=C_GREY, lw=2, label=f"old box model retrodictive ({OLD_RMSE})")
    for xi, v in zip(x, folds.wins_rmse):
        ax.text(xi, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{s-1}-{str(s)[2:]}" for s in folds.season])
    ax.set(ylabel="Wins RMSE", title="Out-of-sample win-prediction error by season")
    ax.legend(); ax.grid(alpha=.25, axis="y")
    plt.tight_layout(); plt.savefig(HERE / "chart_wins_rmse_by_year.png", dpi=110); plt.close(fig)
    print("wrote charts_win_prediction.pdf + chart_wins_rmse_by_year.png")


# ---------------------------------------------------------------------------
# C) player rank charts
# ---------------------------------------------------------------------------
def _bump(ax, allr, top_ids, name_of, cmap):
    for i, pid in enumerate(top_ids):
        d = allr[allr.PLAYER_ID == pid].sort_values("season")
        d = d[d["rank"] <= 30]
        ax.plot(d.season, d["rank"], "-o", color=cmap(i % 10), lw=2.2, ms=6,
                label=name_of[pid])
    ax.invert_yaxis()
    ax.set(xlabel="Season", ylabel="WAA rank (1 = best)", ylim=[30.5, 0.5],
           title="Player WAA rank trajectories (top 10 by cumulative WAA)")
    ax.set_xticks(SEASONS)
    ax.set_xticklabels([f"{s-1}-{str(s)[2:]}" for s in SEASONS], rotation=45)
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(alpha=.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)


def chart_player_ranks(allr):
    cum = allr.groupby(["PLAYER_ID", "NAME"]).waa_wins.sum().reset_index()
    top_ids = cum.sort_values("waa_wins", ascending=False).head(10).PLAYER_ID.tolist()
    name_of = dict(zip(allr.PLAYER_ID, allr.NAME))
    cmap = get_cmap("tab10")

    with PdfPages(HERE / "charts_player_ranks.pdf") as pdf:
        # ---- bump chart of rank trajectories ---------------------------------
        fig, ax = plt.subplots(figsize=(14, 8))
        _bump(ax, allr, top_ids, name_of, cmap)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- heatmap of WAA wins by year (top 25 cumulative) -----------------
        top25 = cum.sort_values("waa_wins", ascending=False).head(25).PLAYER_ID.tolist()
        mat = np.full((len(top25), len(SEASONS)), np.nan)
        for r, pid in enumerate(top25):
            d = allr[allr.PLAYER_ID == pid].set_index("season")
            for c, s in enumerate(SEASONS):
                if s in d.index:
                    mat[r, c] = d.loc[s, "waa_wins"]
        fig, ax = plt.subplots(figsize=(13, 10))
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=18)
        ax.set_xticks(range(len(SEASONS)))
        ax.set_xticklabels([f"{s-1}-{str(s)[2:]}" for s in SEASONS], rotation=45)
        ax.set_yticks(range(len(top25)))
        ax.set_yticklabels([name_of[p] for p in top25], fontsize=9)
        for r in range(len(top25)):
            for c in range(len(SEASONS)):
                if not np.isnan(mat[r, c]):
                    ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_title("WAA wins by season (top 25 players by cumulative WAA)")
        plt.colorbar(im, label="WAA wins"); plt.tight_layout()
        pdf.savefig(fig); fig.savefig(HERE / "chart_player_waa_heatmap.png", dpi=110)
        plt.close(fig)

        # ---- small multiples: top-10 each season -----------------------------
        ncol = 4; nrow = int(np.ceil(len(SEASONS) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.4 * nrow))
        axes = np.array(axes).reshape(-1)
        for ax, s in zip(axes, SEASONS):
            d = allr[allr.season == s].nsmallest(10, "rank").iloc[::-1]
            ax.barh(d.NAME + " (" + d.team + ")", d.waa_wins, color=C_GOOD)
            ax.set_title(f"{s-1}-{str(s)[2:]} top 10 WAA", fontsize=10)
            ax.tick_params(labelsize=7); ax.grid(alpha=.2, axis="x")
        for ax in axes[len(SEASONS):]:
            ax.axis("off")
        fig.suptitle("Top 10 players by WAA wins, each season", fontsize=15, y=1.0)
        plt.tight_layout(); pdf.savefig(fig)
        fig.savefig(HERE / "chart_top10_by_year.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # standalone bump PNG (clean version)
    fig, ax = plt.subplots(figsize=(14, 8))
    _bump(ax, allr, top_ids, name_of, cmap)
    plt.savefig(HERE / "chart_player_rank_bump.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote charts_player_ranks.pdf + PNGs (bump, heatmap, top10_by_year)")


if __name__ == "__main__":
    allr = build_all_ratings()
    chart_win_performance()
    chart_player_ranks(allr)
    print("\nTop 10 by WAA wins, 2024-25:")
    print(allr[allr.season == 2025].nsmallest(10, "rank")[
        ["rank", "NAME", "team", "minutes" if "minutes" in allr else "MINUTES", "waa_wins"]
    ].to_string(index=False))
