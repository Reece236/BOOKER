"""Diagnostic plots for the contract-value model."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from forecast import contract_value as cv

HERE = Path(__file__).resolve().parent
OUT = HERE / "diagnostics_contract.pdf"
POS_MAP = {"G": "PG", "W": "SF", "F": "PF", "C": "C"}
COL = {"ink": "#241c12", "accent": "#7a2820", "pos": "#2f5d34", "neg": "#7a2820", "grid": "#ddcca8"}


def _eval_frame(df):
    rows = []
    for r in df.itertuples():
        final = cv.predict_fair_aav_2026(r.age, POS_MAP[r.pg], r.waa, r.yrs)
        rows.append({
            "pg": r.pg, "source": r.source, "waa": r.waa, "age": r.age,
            "actual": r.aav_2026, "pred": final, "nm": r.nm,
        })
    return pd.DataFrame(rows)


def write_diagnostics(path=OUT):
    cv.fit_linear_waa_model({})
    _, _, df = cv.fit_model({})
    ev = _eval_frame(df)
    fa = ev[ev.source == "fa"].copy()
    fa["log_err"] = np.log(fa.actual) - np.log(fa.pred.clip(1))

    with PdfPages(path) as pdf:
        # 1. Predicted vs actual (FA signings)
        fig, ax = plt.subplots(figsize=(8, 8))
        for pg, color in zip("GWFC", [COL["accent"], COL["ink"], COL["pos"], COL["neg"]]):
            sub = fa[fa.pg == pg]
            ax.scatter(sub.actual / 1e6, sub.pred / 1e6, alpha=0.55, s=28,
                       label=pg, c=color, edgecolors="w", linewidths=0.3)
        lim = [0, max(fa.actual.max(), fa.pred.max()) / 1e6 * 1.05]
        ax.plot(lim, lim, "--", color="gray", lw=1.5)
        ax.set(xlabel="Actual AAV (2026$M)", ylabel="Model fair AAV (2026$M)",
               title="Free-agent signings: fair value vs market", xlim=lim, ylim=lim)
        ax.legend(title="Pos group")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Log residual by position (FA)
        fig, ax = plt.subplots(figsize=(9, 5))
        order = ["G", "W", "F", "C"]
        data = [fa.loc[fa.pg == pg, "log_err"].values for pg in order]
        ax.boxplot(data, labels=order)
        ax.axhline(0, color="gray", ls="--")
        ax.set(ylabel="log(actual) − log(predicted)", title="FA calibration error by position (>0 = model too low)")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. Residual vs WAA by position
        fig, ax = plt.subplots(figsize=(9, 5))
        for pg, color in zip("GWFC", [COL["accent"], COL["ink"], COL["pos"], COL["neg"]]):
            sub = fa[fa.pg == pg]
            ax.scatter(sub.waa, sub.log_err, alpha=0.45, s=22, label=pg, c=color)
        ax.axhline(0, color="gray", ls="--")
        ax.set(xlabel="WAA wins", ylabel="log residual", title="Calibration vs player quality")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 4. Position median ratio (actual / pred) by WAA tier
        fa["tier"] = pd.cut(fa.waa, bins=[-np.inf, 1, 3, 6, np.inf],
                            labels=["<1", "1-3", "3-6", "6+"])
        pivot = fa.groupby(["pg", "tier"], observed=True).apply(
            lambda g: np.median(g.actual / g.pred.clip(1)), include_groups=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(order))
        w = 0.2
        tiers = ["<1", "1-3", "3-6", "6+"]
        for i, tier in enumerate(tiers):
            vals = [pivot.get((pg, tier), np.nan) for pg in order]
            ax.bar(x + (i - 1.5) * w, vals, width=w, label=tier)
        ax.axhline(1.0, color="gray", ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set(ylabel="Median actual / predicted", title="Position pay bias by WAA tier (FA only)")
        ax.legend(title="WAA tier")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 5. Largest under-predictions (stars)
        fa["gap"] = fa.actual - fa.pred
        top = fa.nlargest(20, "gap")
        fig, ax = plt.subplots(figsize=(10, 7))
        labels = top.nm.str[:18] + " (" + top.pg + ")"
        ax.barh(range(len(top)), top.gap / 1e6, color=COL["pos"])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Actual − fair AAV ($M)")
        ax.set_title("Largest FA under-predictions (model fair value too low)")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    write_diagnostics()
