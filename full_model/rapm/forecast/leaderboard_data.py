"""Leaderboard payloads: Bayesian/enhanced WAA, contracts, 2027 projection grades."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import contract_value as cv
from . import enhanced_impacts as ei
from . import player_impacts as pi

HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / "cache"
BAY = CACHE / "bayesian"

PROJ_SEASON = 2027
MIN_PROJ_MINUTES = 200

GRADE_THRESHOLDS = (
    (98, "A+"), (92, "A"), (85, "A-"), (75, "B+"), (62, "B"), (50, "B-"),
    (38, "C+"), (25, "C"), (12, "C-"), (4, "D"), (0, "F"),
)


def _grade_from_pct(pct):
    for thresh, letter in GRADE_THRESHOLDS:
        if pct >= thresh:
            return letter
    return "F"


def _bayesian_rows_for_season(data, season):
    path = BAY / f"player_posterior_{season}.csv"
    if not path.exists() or season not in data.PLAYERS:
        return []
    post = pd.read_csv(path)
    pl = data.PLAYERS[season].copy()
    tmin = pl.groupby("TEAM_ID").MINUTES.transform("sum")
    pl["presence"] = pl.MINUTES / (tmin / 5.0)
    train = pi.prior_train_seasons(data, season)
    if not train:
        return []
    k, _ = pi.fit_net_to_wins(data, train)
    teams = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ABBR))
    rows = []
    pmap = post.set_index("PLAYER_ID")
    for r in pl.itertuples():
        if r.PLAYER_ID not in pmap.index or r.MINUTES < 1:
            continue
        p = pmap.loc[r.PLAYER_ID]
        pres = float(r.presence)
        rows.append({
            "pid": int(r.PLAYER_ID),
            "season": season,
            "waaOff": round(k * float(p.impact_off) * pres, 2),
            "waaDef": round(k * float(p.impact_def) * pres, 2),
            "waaModel": round(k * float(p.impact_total) * pres, 2),
            "waaOff100": round(float(p.impact_off), 2),
            "waaDef100": round(float(p.impact_def), 2),
            "waaModel100": round(float(p.impact_total), 2),
            "sdOff": round(float(p.sd_off), 2),
            "sdDef": round(float(p.sd_def), 2),
            "modelType": "bayesian",
            "team": teams.get(r.TEAM_ID, "?"),
            "minutes": int(r.MINUTES),
        })
    return rows


def load_model_waa_map(data=None):
    """
    (pid, season) -> model WAA fields.

    Primary WAA = enhanced ridge + teammate-fit (stable, box-anchored).
    Bayesian posteriors are attached as waaBayesian / uncertainty only — they
    are not used for ranking after validation showed systematic over-rating of
    secondary scorers and under-rating of primary bigs/creators.
    """
    if data is None:
        data = pi.BookerData(seasons=range(2015, 2028))

    out = {}
    enh_path = HERE / "booker_waa_enhanced_ratings.csv"
    if enh_path.exists():
        df = pd.read_csv(enh_path)
        for r in df.itertuples():
            key = (int(r.pid), int(r.season))
            out[key] = {
                "pid": int(r.pid),
                "season": int(r.season),
                "waaOff": round(float(r.waa_off), 2),
                "waaDef": round(float(r.waa_def), 2),
                "waaModel": round(float(r.waa_total), 2),
                "waaOff100": round(float(r.impact_off), 2),
                "waaDef100": round(float(r.impact_def), 2),
                "waaModel100": round(float(r.impact_total), 2),
                "waa32": round(float(getattr(r, "waa32_total", np.nan)), 2)
                if hasattr(r, "waa32_total") else None,
                "waa32Off": round(float(getattr(r, "waa32_off", np.nan)), 2)
                if hasattr(r, "waa32_off") else None,
                "waa32Def": round(float(getattr(r, "waa32_def", np.nan)), 2)
                if hasattr(r, "waa32_def") else None,
                "rankWaa32": int(r.rank_waa32)
                if hasattr(r, "rank_waa32") and pd.notna(getattr(r, "rank_waa32", np.nan)) else None,
                "modelType": "enhanced",
                "team": r.team,
                "minutes": int(r.minutes),
            }

    bay_unc = {}
    for s in data.seasons:
        for row in _bayesian_rows_for_season(data, s):
            bay_unc[(row["pid"], row["season"])] = row

    # Bayesian-posterior overlay (uncertainty + a comparison WAA). Prefer the
    # BookerFormer ratings -- a variational Bayesian RAPM that beats ridge on OOS
    # team net/wins and produces calibrated, minutes-narrowing per-player sd -- and
    # fall back to the legacy PyMC ratings. Both share the same column schema.
    bf_all = HERE / "booker_bookerformer_ratings.csv"
    bay_all = HERE / "booker_bayesian_ratings.csv"
    unc_path = bf_all if bf_all.exists() else bay_all
    unc_label = "BookerFormer" if unc_path is bf_all else "PyMC"
    if unc_path.exists():
        df = pd.read_csv(unc_path)
        for r in df.itertuples():
            key = (int(r.PLAYER_ID), int(r.season))
            bay_unc[key] = {
                "pid": int(r.PLAYER_ID),
                "season": int(r.season),
                "uncModel": unc_label,
                "waaBayesian": round(float(r.waa_total), 2),
                "waaBayesianOff": round(float(r.waa_off), 2),
                "waaBayesianDef": round(float(r.waa_def), 2),
                "waaBayesian100": round(float(r.impact_total), 2),
                "waaBayesianOff100": round(float(r.impact_off), 2),
                "waaBayesianDef100": round(float(r.impact_def), 2),
                "bookerScore": round(float(getattr(r, "booker_score", np.nan)), 2)
                if hasattr(r, "booker_score") and pd.notna(getattr(r, "booker_score", np.nan)) else None,
                "bookerOff": round(float(getattr(r, "booker_off", np.nan)), 2)
                if hasattr(r, "booker_off") and pd.notna(getattr(r, "booker_off", np.nan)) else None,
                "bookerDef": round(float(getattr(r, "booker_def", np.nan)), 2)
                if hasattr(r, "booker_def") and pd.notna(getattr(r, "booker_def", np.nan)) else None,
                "sdOff": round(float(getattr(r, "sd_off", np.nan)), 2)
                if hasattr(r, "sd_off") else None,
                "sdDef": round(float(getattr(r, "sd_def", np.nan)), 2)
                if hasattr(r, "sd_def") else None,
                "team": r.team,
                "minutes": int(r.minutes),
            }

    for key, bay in bay_unc.items():
        btot = bay.get("waaBayesian", bay.get("waaModel"))
        boff = bay.get("waaBayesianOff", bay.get("waaOff"))
        bdef = bay.get("waaBayesianDef", bay.get("waaDef"))
        b100 = bay.get("waaBayesian100", bay.get("waaModel100"))
        extra = {
            "waaBayesian": round(float(btot), 2) if btot is not None else None,
            "waaBayesianOff": round(float(boff), 2) if boff is not None else None,
            "waaBayesianDef": round(float(bdef), 2) if bdef is not None else None,
            "waaBayesian100": round(float(b100), 2) if b100 is not None else None,
            "waaBayesianOff100": bay.get("waaBayesianOff100"),
            "waaBayesianDef100": bay.get("waaBayesianDef100"),
            "bookerScore": bay.get("bookerScore"),
            "bookerOff": bay.get("bookerOff"),
            "bookerDef": bay.get("bookerDef"),
            "uncModel": bay.get("uncModel"),
        }
        if bay.get("sdOff") is not None:
            extra["sdOff"] = bay["sdOff"]
        if bay.get("sdDef") is not None:
            extra["sdDef"] = bay["sdDef"]
        is_bf = bay.get("uncModel") == "BookerFormer" and extra["waaBayesian"] is not None
        if key in out:
            row = out[key]
            row.update(extra)
            # Promote BookerFormer to the PRIMARY ranking model, keeping the prior
            # enhanced (teammate-fit ridge) numbers as a comparison column. Seasons
            # with no BookerFormer rating (pre-2018) keep the enhanced model.
            if is_bf:
                row["waaEnhanced"] = row.get("waaModel")
                row["waaEnhancedOff"] = row.get("waaOff")
                row["waaEnhancedDef"] = row.get("waaDef")
                row["waaEnhanced100"] = row.get("waaModel100")
                row["waaModel"] = extra["waaBayesian"]
                row["waaOff"] = extra["waaBayesianOff"]
                row["waaDef"] = extra["waaBayesianDef"]
                row["waaModel100"] = extra["waaBayesian100"]
                if extra["waaBayesianOff100"] is not None:
                    row["waaOff100"] = extra["waaBayesianOff100"]
                if extra["waaBayesianDef100"] is not None:
                    row["waaDef100"] = extra["waaBayesianDef100"]
                row["modelType"] = "bookerformer"
        else:
            # no enhanced row — BookerFormer (or legacy bayesian) becomes primary
            out[key] = {
                "pid": bay["pid"],
                "season": bay["season"],
                "waaOff": bay.get("waaBayesianOff", bay.get("waaOff")),
                "waaDef": bay.get("waaBayesianDef", bay.get("waaDef")),
                "waaModel": bay.get("waaBayesian", bay.get("waaModel")),
                "waaOff100": bay.get("waaBayesianOff100", bay.get("waaOff100")),
                "waaDef100": bay.get("waaBayesianDef100", bay.get("waaDef100")),
                "waaModel100": bay.get("waaBayesian100"),
                "waaBayesian": bay.get("waaBayesian", bay.get("waaModel")),
                "sdOff": bay.get("sdOff"),
                "sdDef": bay.get("sdDef"),
                "uncModel": bay.get("uncModel"),
                "modelType": "bookerformer" if is_bf else "bayesian",
                "team": bay.get("team"),
                "minutes": bay.get("minutes"),
            }
    return out


def build_2027_projections(data=None):
    """Prior-only enhanced impacts on the cloned 2027 roster -> WAA + letter grade."""
    if data is None:
        data = pi.BookerData(seasons=range(2015, 2028))
    if PROJ_SEASON not in data.PLAYERS:
        return {}

    season = PROJ_SEASON
    train = pi.prior_train_seasons(data, season)
    if not train:
        return {}

    k, _ = pi.fit_net_to_wins(data, train)
    enh = ei.build_enhanced(data, train, season)
    comps = ei.player_waa_components(data, season, k, enh)
    pool = [c for c in comps if c["minutes"] >= MIN_PROJ_MINUTES]
    if not pool:
        return {}

    vals = np.array([c["waa_total"] for c in pool])
    order = np.argsort(-vals)
    n = len(pool)
    out = {}
    for i, idx in enumerate(order):
        c = pool[idx]
        pct = 100.0 * (1.0 - i / max(n - 1, 1))
        out[int(c["pid"])] = {
            "waaProj2027": round(float(c["waa_total"]), 2),
            "waaOffProj2027": round(float(c["waa_off"]), 2),
            "waaDefProj2027": round(float(c["waa_def"]), 2),
            "grade2027": _grade_from_pct(pct),
            "projRank2027": int(i + 1),
        }
    return out


def attach_contract_fields(row, pos_by_nm, ages, yos, booker_for_contract):
    nm = cv.norm_name(row["player"])
    pos = pos_by_nm.get(nm, "SF")
    age = ages.get(nm, 27.0)
    yp = yos.get(nm, 4)
    c = cv.player_contract_row(row["player"], pos, age, booker_for_contract, yp, None)
    row.update({
        "trueValue": c["trueValue"],
        "fairAav2026": c["fairAav2026"],   # alias (now holds True Value)
        "marketAav2026": c["marketAav2026"],
        "surplus": c["surplus"],
        "isKnownDeal": c["isKnownDeal"],
    })
    return row
