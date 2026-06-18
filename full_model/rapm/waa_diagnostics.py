"""Diagnostic plots: Bayesian vs enhanced WAA, position bias, outliers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
OUT = HERE / "diagnostics_waa_model.pdf"
COL = {"ink": "#241c12", "accent": "#7a2820", "pos": "#2f5d34", "neg": "#7a2820", "grid": "#ddcca8"}


def _position_map():
    from forecast import player_impacts as pi
    bk = pd.read_csv(pi.PLAYER_DATA)
    bk["nm"] = bk.playerName.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    return bk.groupby("nm").position.last().to_dict()


def _pos_group(pos):
    p = str(pos).upper()
    if "C" in p and "G" not in p:
        return "C"
    if "G" in p and "F" not in p and "C" not in p:
        return "G"
    if "F" in p:
        return "F"
    return "W"


def build_compare(season=2026):
    bay = pd.read_csv(HERE / "booker_bayesian_ratings.csv")
    enh = pd.read_csv(HERE / "booker_waa_enhanced_ratings.csv")
    b = bay[bay.season == season].rename(columns={"PLAYER_ID": "pid", "waa_total": "bay", "player": "player_bay"})
    e = enh[enh.season == season][["pid", "player", "team", "waa_total", "waa_off", "waa_def"]]
    e = e.rename(columns={"waa_total": "enh", "waa_off": "enh_off", "waa_def": "enh_def"})
    m = b.merge(e, on="pid", how="inner")
    m["delta"] = m.bay - m.enh
    pos_lu = _position_map()
    m["nm"] = m.player.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    m["pg"] = m.nm.map(lambda n: _pos_group(pos_lu.get(n, "SF")))
    return m


def write_diagnostics(season=2026, path=OUT):
    m = build_compare(season)
    with PdfPages(path) as pdf:
        # 1. Scatter enhanced vs bayesian
        fig, ax = plt.subplots(figsize=(8, 8))
        for pg, c in [("G", COL["accent"]), ("W", COL["ink"]), ("F", COL["pos"]), ("C", COL["neg"])]:
            sub = m[m.pg == pg]
            ax.scatter(sub.enh, sub.bay, alpha=0.55, s=28, label=pg, c=c, edgecolors="w", linewidths=0.3)
        lim = [m[["enh", "bay"]].min().min() - 1, m[["enh", "bay"]].max().max() + 1]
        ax.plot(lim, lim, "--", color="gray")
        ax.set(xlabel="Enhanced WAA wins", ylabel="Bayesian WAA wins",
               title=f"{season-1}-{str(season)[2:]} Bayesian vs enhanced (diagonal = agreement)")
        ax.legend(title="Pos")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Residual by position
        fig, ax = plt.subplots(figsize=(8, 5))
        order = ["G", "W", "F", "C"]
        ax.boxplot([m.loc[m.pg == pg, "delta"].values for pg in order], labels=order)
        ax.axhline(0, color="gray", ls="--")
        ax.set(ylabel="Bayesian − enhanced WAA", title="Position bias (positive = Bayesian too high)")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Biggest over-rates (NAW problem)
        fig, ax = plt.subplots(figsize=(10, 7))
        top = m.nlargest(18, "delta")
        ax.barh(range(len(top)), top["delta"].values, color=COL["neg"])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([f"{p} ({pg})" for p, pg in zip(top.player, top.pg)], fontsize=8)
        ax.set_xlabel("Bayesian − enhanced WAA")
        ax.set_title("Largest Bayesian over-ratings")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Biggest under-rates (Jokic / stars)
        fig, ax = plt.subplots(figsize=(10, 7))
        bot = m.nsmallest(18, "delta")
        ax.barh(range(len(bot)), bot["delta"].values, color=COL["pos"])
        ax.set_yticks(range(len(bot)))
        ax.set_yticklabels([f"{p} ({pg})" for p, pg in zip(bot.player, bot.pg)], fontsize=8)
        ax.set_xlabel("Bayesian − enhanced WAA")
        ax.set_title("Largest Bayesian under-ratings (often bigs / primary creators)")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 5. Leaderboard top 25 comparison
        fig, ax = plt.subplots(figsize=(11, 8))
        top_enh = m.nlargest(25, "enh").sort_values("enh")
        y = range(len(top_enh))
        ax.barh([i - 0.2 for i in y], top_enh.enh, height=0.35, label="Enhanced (leaderboard)", color=COL["pos"])
        ax.barh([i + 0.2 for i in y], top_enh.bay, height=0.35, label="Bayesian (experimental)", color=COL["neg"])
        ax.set_yticks(list(y))
        ax.set_yticklabels(top_enh.player, fontsize=8)
        ax.set_xlabel("WAA wins")
        ax.set_title(f"Top 25 by enhanced WAA — {season-1}-{str(season)[2:]}")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 6. Team net overshoot (carry-job diagnostic)
        from forecast import enhanced_impacts as ei
        from forecast import player_impacts as pi
        data = pi.BookerData(seasons=range(2015, 2028))
        train = [s for s in range(season - 3, season) if s >= 2015 and s in data.STINTS]
        enh = ei.build_enhanced(data, train, season)
        _, _, pred = ei.aggregate_off_def(data, enh, season)
        tm = data.TEAMS[season]
        abbr = dict(zip(tm.TEAM_ID, tm.ABBR))
        act = dict(zip(tm.TEAM_ID, tm.ACTUAL_NET))
        tg = [(abbr[t], float(pred[t]), float(act[t]), float(pred[t]) - float(act[t]))
              for t in pred if t in act and pd.notna(act[t])]
        tg.sort(key=lambda x: -x[3])
        fig, ax = plt.subplots(figsize=(10, 6))
        labs = [x[0] for x in tg[:20]]
        gaps = [x[3] for x in tg[:20]]
        ax.barh(range(len(labs)), gaps, color=[COL["neg"] if g > 0 else COL["pos"] for g in gaps])
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=8)
        ax.axvline(0, color="gray", ls="--")
        ax.set_xlabel("Predicted roster net − actual team net")
        ax.set_title("Team-level overshoot (positive = inflated roster / carry job)")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"wrote {path}")
    pg_med = m.groupby("pg")["delta"].median()
    print("Median bayesian − enhanced by position:")
    print(pg_med.to_string())
    return path


if __name__ == "__main__":
    write_diagnostics()
