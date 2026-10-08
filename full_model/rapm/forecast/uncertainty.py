"""
Heteroscedastic / hierarchical uncertainty for BOOKER.

Two deliverables, both keeping the POINT estimates untouched:

1. player_sd(df) -- a per-player posterior rating std that actually VARIES with the
   things it should: sample size (minutes), career-trajectory uncertainty (age, a
   U-shape -- rookies AND aging players wider), role (position; interior-defense and
   high-usage scoring value are noisier to attribute), and usage-driven scoring
   volatility. Shape is the textbook standard-error curve SE ~ sqrt(A/n + floor^2):
   the 1/n term makes low-minute players (Embiid, R.Williams) genuinely wider than
   high-minute anchors (Jokic), fixing the old near-constant sd (CV 0.08, corr with
   minutes only -0.19). The single global prior TAU_RATING=2.0 used to cap every
   posterior at ~2.0 regardless of minutes; this replaces that with a per-player
   width.

2. team_net_sd(...) + the per-sim net shock used by preseason.simulate -- the old
   win-band sim only had independent per-game coin-flip (binomial) variance, so a
   team's TRUE net was treated as known. That covers ~50% of outcomes (target 80%).
   Real bands must add a *correlated* net-rating shock (rating uncertainty + roster
   fragility/availability + systematic net->wins error), which moves all 82 games
   together and roughly doubles the win-total sd to match the observed 8.8-win RMSE.
   A single calibrated NET_SCALE hits 80% coverage; the per-team shape is roster-driven.

No torch here -- this runs as a fast post-process over the ratings CSV and inside the
(numpy) preseason sim.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import player_impacts as pi

MASTER = pi.ROOT / "full_model" / "nba_master_dataset_with_archetypes.csv"

# --- player-sd curve (per side, points/100) --------------------------------
# SE ~ sqrt(A_MIN / minutes + B2_FLOOR); calibrated so median sd_total ~2.1 and the
# minutes slope is strong (corr(sd,minutes) ~ -0.65). B2_FLOOR is the irreducible
# per-side variance a fully-observed player still carries.
A_MIN = 1388.0
B2_FLOOR = 1.03
M_FLOOR = 200.0        # minutes floor so tiny samples don't diverge
SD_CAL = 1.0           # global level calibration (raise to widen all player bands)

POS_OFF_MULT = {"PG": 1.05, "SG": 1.02, "SF": 1.00, "PF": 0.99, "C": 0.97}
POS_DEF_MULT = {"PG": 0.98, "SG": 0.99, "SF": 1.00, "PF": 1.06, "C": 1.10}


def _age_mult(age):
    """U-shape: prime (~27) tightest, rookies and aging players wider."""
    a = np.where(np.isnan(age), 27.0, age)
    return np.clip(0.96 + 0.004 * (a - 27.0) ** 2, 0.94, 1.30)


def _usage_off_mult(usage):
    """High-usage scorers are more boom/bust -> wider offensive band."""
    u = np.where(np.isnan(usage), 20.0, usage)
    return np.clip(1.0 + 0.010 * (u - 22.0), 0.90, 1.18)


_MASTER_CACHE = {}


def _master_attrs():
    """{norm_name: (ref_season, age, position, usage)} newest-season reference row,
    for carry-forward. Master ends ~2025; older/newer target seasons extrapolate age."""
    if "attrs" in _MASTER_CACHE:
        return _MASTER_CACHE["attrs"]
    attrs = {}
    if MASTER.exists():
        m = pd.read_csv(MASTER, usecols=["playerName", "season", "age",
                                         "position", "usagePercent"])
        m["nm"] = m.playerName.map(pi.norm_name)
        m["season"] = pd.to_numeric(m.season, errors="coerce")
        m = m.dropna(subset=["season"]).sort_values("season")
        for r in m.itertuples():
            attrs[r.nm] = (int(r.season), r.age, r.position, r.usagePercent)  # last wins = newest
    _MASTER_CACHE["attrs"] = attrs
    return attrs


def attach_attrs(df):
    """Add age/position/usage columns to a ratings frame (needs `player`,`season`),
    carrying the newest master row forward and aging players by the season gap."""
    attrs = _master_attrs()
    ages, poss, uses = [], [], []
    for nm, s in zip(df.player.map(pi.norm_name), df.season):
        a = attrs.get(nm)
        if a is None:
            ages.append(np.nan); poss.append("SF"); uses.append(np.nan)
        else:
            ref_s, age, pos, use = a
            ages.append((age + (int(s) - ref_s)) if pd.notna(age) else np.nan)
            poss.append(pos if isinstance(pos, str) else "SF")
            uses.append(use)
    out = df.copy()
    out["age"] = np.array(ages, dtype=float)
    out["position"] = poss
    out["usage"] = np.array(uses, dtype=float)
    return out


def player_sd(df):
    """Return (sd_off, sd_def) arrays for a ratings frame with `minutes` and, ideally,
    `age`/`position`/`usage` (call attach_attrs first). Point estimates untouched."""
    d = df if "age" in df.columns else attach_attrs(df)
    minutes = np.clip(d.minutes.to_numpy(dtype=float), M_FLOOR, None)
    base = np.sqrt(A_MIN / minutes + B2_FLOOR)          # per-side SE curve
    am = _age_mult(d.age.to_numpy(dtype=float))
    po = np.array([POS_OFF_MULT.get(p, 1.0) for p in d.position])
    pd_ = np.array([POS_DEF_MULT.get(p, 1.0) for p in d.position])
    uo = _usage_off_mult(d.usage.to_numpy(dtype=float))
    sd_off = SD_CAL * base * am * po * uo
    sd_def = SD_CAL * base * am * pd_
    return sd_off, sd_def


def rewrite_ratings_sd(path):
    """One-off / post-fit: recompute sd_off/sd_def in a ratings CSV in place
    (heteroscedastic), leaving every point-estimate column untouched."""
    df = pd.read_csv(path)
    df = attach_attrs(df)
    so, sd = player_sd(df)
    df["sd_off"] = np.round(so, 3)
    df["sd_def"] = np.round(sd, 3)
    df = df.drop(columns=["age", "position", "usage"])
    df.to_csv(path, index=False)
    return df


# --- team win-band uncertainty --------------------------------------------
# Per-team correlated net shock the sim adds. Two components:
#   * rating_var  -- rolls up player-rating uncertainty (RHO-tightened for negative
#     teammate correlation). This is the one roster-driven term that is empirically
#     (if weakly) predictive of win-total error: teams leaning on high-uncertainty
#     (low-minute / injury-limited) players miss projections by more (7.2 vs 5.8 wins,
#     top vs bottom rating_var tercile over 270 team-seasons).
#   * RESID_VAR   -- a systematic floor. This DOMINATES, because win-total misses are
#     mostly unpredictable (breakouts, trades, health) and do NOT correlate with
#     roster shape: impact-concentration/fragility was tested and had the WRONG sign
#     (concentrated teams missed slightly *less*, corr -0.05), so it is deliberately
#     omitted -- team win-uncertainty is genuinely ~homogeneous. The real, predictable
#     heterogeneity lives in the PLAYER bands (player_sd), not the team bands.
# NET_SCALE is the single knob calibrated to 80% coverage.
RHO = 0.78             # joint/marginal team-net variance ratio (neg teammate corr)
RESID_VAR = 1.0        # systematic net floor (net->wins + irreducible), (pts/100)^2
NET_SCALE = 1.0        # global calibration -> 80% coverage (set by calibrate_net_scale)


def team_net_sd(ratings, season, net_scale=None):
    """{team_abbr: net-rating std} for a season, from that season's roster in the
    ratings CSV: RHO-tightened player-rating covariance + a systematic floor."""
    scale = NET_SCALE if net_scale is None else net_scale
    d = ratings[ratings.season == season]
    out = {}
    for team, g in d.groupby("team"):
        tot_min = g.minutes.sum()
        if tot_min <= 0:
            continue
        pres = g.minutes.to_numpy(float) / (tot_min / 5.0)      # sums to ~5
        sd_tot = np.hypot(g.sd_off.to_numpy(float), g.sd_def.to_numpy(float))
        w = pres / 5.0                                          # per-player net weight
        rating_var = RHO * float(np.sum((w * sd_tot) ** 2))
        out[team] = scale * np.sqrt(rating_var + RESID_VAR)
    return out
