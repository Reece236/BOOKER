"""
NBA salary inflation + fair-market contract valuation in target-year dollars.

Fair AAV = linear WAA -> market-value fit on FA signings (2026$), uncapped.
Surplus = fair AAV minus the player's actual contract (when known).

Data:
  * NBA Player Salaries_2000-2025.csv (local)
  * 2026_sals.xls — Basketball-Reference HTML export (2025-26 salaries)
  * Free Agent Signings - Sheet1.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

from . import player_impacts as pi

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
FA_PATH = RAPM / "Free Agent Signings - Sheet1.csv"
SALARY_HIST_PATH = RAPM / "NBA Player Salaries_2000-2025.csv"
SALARY_2026_PATH = RAPM / "2026_sals.xls"
MODEL_CACHE = CACHE / "contract_value_model.joblib"
LINEAR_CACHE = CACHE / "contract_linear_waa.joblib"
TRUEVALUE_CACHE = CACHE / "truevalue_model.joblib"
BOOKER_RATINGS = RAPM / "booker_bookerformer_ratings.csv"

TARGET_YEAR = 2026

CAP_BY_SEASON = {
    2000: 34.5e6, 2005: 49.5e6, 2010: 58.04e6, 2015: 70.0e6,
    2016: 94.14e6, 2017: 99.09e6, 2018: 101.87e6, 2019: 109.14e6,
    2020: 109.14e6, 2021: 112.41e6, 2022: 123.66e6, 2023: 136.02e6,
    2024: 140.59e6, 2025: 154.65e6, 2026: 154.65e6,
}
TAX_LINE_2026 = 187.895e6
APR1_2026 = 178.132e6
APR2_2026 = 188.931e6
MLE_2026 = 15.672e6
TAXPAYER_MLE_2026 = 6.183e6

VET_MIN_2026 = {
    0: 1_193_057, 1: 1_193_057, 2: 1_955_377, 3: 2_399_537,
    4: 2_486_452, 5: 2_573_366, 6: 2_660_281, 7: 2_747_195,
    8: 2_834_110, 9: 2_921_024, 10: 3_007_938,
}

POS_GROUP = {
    "PG": "G", "SG": "G", "G": "G", "G/F": "W", "SF": "W", "SG/SF": "W",
    "SF/PF": "W", "PF/SF": "W", "PF": "F", "F": "F", "PF/PF": "F",
    "C/PF": "C", "C": "C", "F/C": "C",
}

# FA-calibrated premia for rotation+ starters (applied when WAA >= 2)
POS_WAA_MULT = {"G": 0.94, "W": 1.02, "F": 1.16, "C": 1.22}

STAR_FLOOR_PCT = {"G": 0.27, "W": 0.28, "F": 0.30, "C": 0.32}


def norm_name(s):
    return pi.norm_name(s)


def parse_money(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = re.sub(r"[^0-9.]", "", str(s))
    return float(t) if t else np.nan


def cap_for_season(season):
    if season in CAP_BY_SEASON:
        return CAP_BY_SEASON[season]
    years = sorted(CAP_BY_SEASON)
    if season <= years[0]:
        return CAP_BY_SEASON[years[0]]
    if season >= years[-1]:
        return CAP_BY_SEASON[years[-1]]
    xs = np.array(years, dtype=float)
    ys = np.log(np.array([CAP_BY_SEASON[y] for y in years]))
    return float(np.exp(np.interp(season, xs, ys)))


def inflation_factor(from_season, to_season=TARGET_YEAR):
    return cap_for_season(to_season) / cap_for_season(from_season)


def inflate_salary(amount, from_season, to_season=TARGET_YEAR):
    if amount is None or np.isnan(amount):
        return np.nan
    return float(amount) * inflation_factor(from_season, to_season)


def _position_lookup():
    bk = pd.read_csv(pi.PLAYER_DATA)
    bk["nm"] = bk.playerName.map(norm_name)
    pos = {}
    for (nm, ss), p in bk.groupby(["nm", "season"]).position.last().items():
        pos[(nm, int(ss))] = str(p).strip()
    latest = bk.sort_values("season").groupby("nm").position.last().to_dict()
    return pos, latest


def load_2026_salaries():
    """Parse BBR HTML salary table (2025-26 season -> end-year 2026)."""
    if not SALARY_2026_PATH.exists():
        return pd.DataFrame(columns=["name", "salary", "season", "nm", "team"])
    t = pd.read_html(SALARY_2026_PATH)[0]
    player = t.iloc[:, 1].astype(str)
    team = t.iloc[:, 2].astype(str)
    sal = t.iloc[:, 3].map(parse_money)
    out = pd.DataFrame({
        "name": player,
        "team": team,
        "salary": sal,
        "season": TARGET_YEAR,
    })
    out = out[out.salary.notna() & (out.salary > 0)].copy()
    out["nm"] = out.name.map(norm_name)
    return out


def load_salary_history():
    if not SALARY_HIST_PATH.exists():
        raise FileNotFoundError(f"missing salary file: {SALARY_HIST_PATH}")
    df = pd.read_csv(SALARY_HIST_PATH)
    df = df.rename(columns={"Player": "name", "Salary": "salary", "Season": "season"})
    df["season"] = pd.to_numeric(df.season, errors="coerce").astype("Int64")
    df["salary"] = pd.to_numeric(df.salary, errors="coerce")
    df["nm"] = df.name.map(norm_name)
    df = df[df.salary.notna() & (df.salary > 0)].copy()

    s26 = load_2026_salaries()
    if not s26.empty:
        df = pd.concat([
            df,
            s26[["name", "salary", "season", "nm"]].assign(team=s26.get("team")),
        ], ignore_index=True)

    pos_ss, pos_latest = _position_lookup()
    fa = load_fa_signings()
    fa_pos = fa.sort_values("sign_year").groupby("nm").POS.last().to_dict()

    def _pg(row):
        p = pos_ss.get((row.nm, int(row.season)), fa_pos.get(row.nm, pos_latest.get(row.nm, "SF")))
        return POS_GROUP.get(str(p).strip(), "W")

    df["pg"] = df.apply(_pg, axis=1)
    return df


def load_fa_signings():
    df = pd.read_csv(FA_PATH)
    df = df.rename(columns={"PLAYER (128)": "player", "Year": "sign_year"})
    df["nm"] = df.player.map(norm_name)
    df["aav"] = df.AAV.map(parse_money)
    df["yrs"] = pd.to_numeric(df.YRS, errors="coerce")
    df["pg"] = df.POS.map(lambda p: POS_GROUP.get(str(p).strip(), "W"))
    df["season"] = df.sign_year + 1
    df["aav_2026"] = df.apply(
        lambda r: inflate_salary(r.aav, int(r.season), TARGET_YEAR), axis=1)
    return df


def _player_ages():
    bk = pd.read_csv(pi.PLAYER_DATA)
    bk["nm"] = bk.playerName.map(norm_name)
    ages = {}
    for (nm, ss), g in bk.groupby(["nm", "season"]):
        ages[(nm, int(ss))] = float(np.average(g.age, weights=g.minutesPlayed.clip(1)))
    latest = bk.sort_values("season").groupby("nm").tail(1).set_index("nm")
    return ages, latest


def build_waa_by_season():
    out = {}
    bay = RAPM / "booker_bayesian_ratings.csv"
    if bay.exists():
        df = pd.read_csv(bay)
        for r in df.itertuples():
            out[(norm_name(r.player), int(r.season))] = float(r.waa_total)
    path = RAPM / "booker_waa_ratings_by_year.csv"
    if path.exists():
        df = pd.read_csv(path)
        for r in df.itertuples():
            key = (norm_name(r.player), int(r.season))
            out.setdefault(key, float(r.waa_wins))
    return out


def build_training_frame(waa_by_name=None):
    fa = load_fa_signings()
    sal = load_salary_history()
    ages, latest_age = _player_ages()
    waa_ss = build_waa_by_season()
    rows = []

    def _waa(nm, season):
        if (nm, season) in waa_ss:
            return waa_ss[(nm, season)]
        return (waa_by_name or {}).get(nm, 0.0)

    for r in fa.itertuples():
        age = ages.get((r.nm, int(r.season)), np.nan)
        if np.isnan(age) and r.nm in latest_age.index:
            age = float(latest_age.loc[r.nm, "age"])
        if np.isnan(age):
            age = 27.0
        rows.append({
            "nm": r.nm, "aav_2026": r.aav_2026, "age": age, "pg": r.pg,
            "yrs": int(r.yrs) if not np.isnan(r.yrs) else 2,
            "waa": _waa(r.nm, int(r.season)), "source": "fa",
        })

    # High-salary seasons supplement FA (star structure) without flooding with minimum deals
    for r in sal.itertuples():
        if r.salary < 8_000_000 or int(r.season) < 2018:
            continue
        aav_inf = inflate_salary(r.salary, int(r.season), TARGET_YEAR)
        age = ages.get((r.nm, int(r.season)), np.nan)
        if np.isnan(age):
            age = 28.0
        rows.append({
            "nm": r.nm, "aav_2026": aav_inf, "age": age, "pg": r.pg,
            "yrs": 3, "waa": _waa(r.nm, int(r.season)), "source": "salary",
        })

    return pd.DataFrame(rows)


def _age_features(age):
    age = np.clip(age, 19, 39)
    return np.column_stack([
        age, age ** 2, np.maximum(0, age - 27), np.maximum(0, 27 - age),
    ])


def _feature_matrix(age, pg, waa, yrs, enc):
    age = np.clip(float(age), 19, 39)
    waa = max(float(waa), 0.0)
    yrs = int(yrs)
    age_f = _age_features(np.array([age]))
    pg_o = enc.transform(np.array([[pg]]))
    is_g = float(pg == "G")
    is_w = float(pg == "W")
    is_f = float(pg == "F")
    is_c = float(pg == "C")
    return np.hstack([
        age_f, pg_o,
        np.array([[np.log1p(waa)]]),
        np.array([[np.sqrt(waa)]]),
        np.array([[yrs]]),
        np.array([[waa * is_g]]),
        np.array([[waa * is_w]]),
        np.array([[waa * is_f]]),
        np.array([[waa * is_c]]),
    ])


def _gbm_predict(age, pg, waa, years_pro, model, enc):
    X = _feature_matrix(age, pg, waa, years_pro, enc)
    return float(np.exp(model.predict(X)[0]))


def fit_model(waa_by_name=None):
    df = build_training_frame(waa_by_name)
    df = df[df.aav_2026 > 500_000].copy()
    df["waa"] = df.waa.fillna(0.0)
    df["age"] = df.age.fillna(27.0)
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc.fit(df[["pg"]])
    X = np.vstack([
        _feature_matrix(r.age, r.pg, r.waa, r.yrs, enc)
        for r in df.itertuples()
    ])
    y = np.log(df.aav_2026.values)
    weights = np.where(df.source.values == "fa", 12.0, 2.0)
    model = GradientBoostingRegressor(
        n_estimators=220, max_depth=4, learning_rate=0.05,
        subsample=0.85, random_state=42,
    )
    model.fit(X, y, sample_weight=weights)
    import joblib
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "enc": enc}, MODEL_CACHE)
    return model, enc, df


def _load_model(waa_by_name=None):
    if MODEL_CACHE.exists():
        import joblib
        blob = joblib.load(MODEL_CACHE)
        return blob["model"], blob["enc"]
    return fit_model(waa_by_name)[:2]


def cba_clamp(aav_2026, age, years_pro, waa_total):
    cap = cap_for_season(TARGET_YEAR)
    yos = int(np.clip(years_pro, 0, 10))
    floor = VET_MIN_2026.get(yos, VET_MIN_2026[10])
    aav_2026 = max(aav_2026, floor)

    if waa_total >= 6.0:
        mx = 0.35 * cap
    elif waa_total >= 3.5:
        mx = 0.30 * cap
    elif waa_total >= 1.5:
        mx = 0.25 * cap
    else:
        mx = MLE_2026
    if waa_total < 0.5:
        mx = min(mx, TAXPAYER_MLE_2026 * 1.15)
    return float(min(aav_2026, mx))


def fit_linear_waa_model(waa_by_name=None):
    """OLS fair AAV = intercept + slope * WAA on FA signings (2026 dollars)."""
    fa = load_fa_signings()
    waa_ss = build_waa_by_season()
    rows = []
    for r in fa.itertuples():
        waa = waa_ss.get((r.nm, int(r.season)))
        if waa is None:
            waa = (waa_by_name or {}).get(r.nm, 0.0)
        if r.aav_2026 is None or np.isnan(r.aav_2026) or r.aav_2026 < 500_000:
            continue
        rows.append({"waa": float(waa), "aav_2026": float(r.aav_2026)})
    df = pd.DataFrame(rows)
    if df.empty:
        intercept, slope = 2.27e6, 1.70e6
    else:
        slope, intercept = np.polyfit(df.waa.values, df.aav_2026.values, 1)
    blob = {"intercept": float(intercept), "slope": float(slope), "n_fa": len(df)}
    import joblib
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(blob, LINEAR_CACHE)
    return blob


def _load_linear_model(waa_by_name=None):
    if LINEAR_CACHE.exists():
        import joblib
        return joblib.load(LINEAR_CACHE)
    return fit_linear_waa_model(waa_by_name)


def predict_fair_aav_2026(age, pos, waa_total, years_pro=4, waa_by_name=None):
    """Uncapped linear fair value from WAA (2026 dollars). Legacy; superseded by
    True Value (predict_true_value)."""
    lin = _load_linear_model(waa_by_name)
    waa = float(waa_total) if waa_total is not None and not (
        isinstance(waa_total, float) and np.isnan(waa_total)) else 0.0
    return float(lin["intercept"] + lin["slope"] * waa)


# ---------------------------------------------------------------------------
# True Value: AAV ~ b0 + b1*BOOKER(prior season) + b2*max(0, age-29), fit on FA
# signings. Prediction zeroes the age term -> the salary a player's SKILL alone
# (BOOKER score) commands, stripping the market's age discount.
# ---------------------------------------------------------------------------
def build_booker_by_season():
    """(nm, season) -> BOOKER score (predictive WAA / 3000 poss)."""
    out = {}
    if BOOKER_RATINGS.exists():
        df = pd.read_csv(BOOKER_RATINGS)
        if "booker_score" in df.columns:
            for r in df.itertuples():
                if pd.notna(getattr(r, "booker_score", np.nan)):
                    out[(norm_name(r.player), int(r.season))] = float(r.booker_score)
    return out


# True Value is read off the fitted surface at a young-prime reference age, which
# strips the market's age discount (skill alone). The model is monotone increasing
# in BOOKER and decreasing in age, so: better BOOKER is always worth more at a given
# age, and a younger player is always worth more at a given BOOKER.
TRUE_VALUE_REF_AGE = 25.0


def fit_truevalue_model():
    """Monotonic gradient-boosted True-Value model. Features [BOOKER(season before
    FA), age]; target log AAV (2026$). Monotone constraints: BOOKER +1, age -1.
    Cached to TRUEVALUE_CACHE."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    fa = load_fa_signings()
    booker = build_booker_by_season()
    ages, latest_age = _player_ages()

    def _age(nm, season):
        a = ages.get((nm, int(season)))
        if a is None and nm in latest_age.index:
            a = float(latest_age.loc[nm, "age"])
        return 27.0 if (a is None or np.isnan(a)) else float(a)

    X, y, w = [], [], []
    for r in fa.itertuples():
        b = booker.get((r.nm, int(r.sign_year)))   # the season just played, pre-FA
        if b is None or r.aav_2026 is None or np.isnan(r.aav_2026) or r.aav_2026 < 500_000:
            continue
        X.append([b, _age(r.nm, r.sign_year)])
        y.append(np.log(float(r.aav_2026)))
        w.append(3.0)   # true FA signings are the primary signal

    # High-BOOKER players rarely reach free agency (they sign extensions), so FA data
    # alone can't anchor the elite tail. Supplement with star-salary player-seasons
    # (BOOKER that season, age, AAV in 2026$) so the monotone fit keeps rising at the
    # top instead of flat-lining; FA rows stay weighted higher.
    sal = load_salary_history()
    for r in sal.itertuples():
        if int(r.season) < 2018 or r.salary is None or np.isnan(r.salary) or r.salary < 15_000_000:
            continue
        b = booker.get((r.nm, int(r.season)))
        if b is None:
            continue
        X.append([b, _age(r.nm, r.season)])
        y.append(np.log(inflate_salary(float(r.salary), int(r.season), TARGET_YEAR)))
        w.append(1.0)
    import joblib
    CACHE.mkdir(parents=True, exist_ok=True)
    n_fa = int(sum(1 for wi in w if wi >= 3.0))
    if len(X) < 20:
        # tiny-sample fallback: monotone-ish linear in BOOKER only
        blob = {"kind": "linear", "b0": 11.0e6, "b1": 3.4e6, "n": len(X),
                "n_fa": n_fa, "ref_age": TRUE_VALUE_REF_AGE}
        joblib.dump(blob, TRUEVALUE_CACHE)
        return blob
    # Additive (interaction_cst keeps BOOKER and age in separate groups, i.e. a
    # monotone GAM): the BOOKER effect is learned from ALL ages -- including older
    # stars with high BOOKER -- so it keeps rising at the top instead of flat-lining
    # in the data-sparse young-and-elite corner, while age stays a separate monotone
    # term we can zero out for True Value.
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_depth=2, max_iter=400, learning_rate=0.04,
        min_samples_leaf=12, l2_regularization=0.3,
        monotonic_cst=[1, -1], interaction_cst=[[0], [1]], random_state=42,
    )
    model.fit(np.array(X), np.array(y), sample_weight=np.array(w))
    # Salaries are right-censored at the supermax, so the fitted surface flattens at
    # the top. A player's VALUE isn't capped, so beyond the dense-data region we
    # extrapolate a straight, uncapped tail at the sub-max market's marginal $/BOOKER
    # rate. booker_q delimits where the tail kicks in (90th pct of training BOOKER).
    bk = np.array([row[0] for row in X])
    booker_q = (float(np.quantile(bk, 0.50)), float(np.quantile(bk, 0.90)))
    blob = {"kind": "gbm", "model": model, "log": True, "n": len(X),
            "n_fa": n_fa, "ref_age": TRUE_VALUE_REF_AGE, "booker_q": booker_q}
    joblib.dump(blob, TRUEVALUE_CACHE)
    return blob


def _load_truevalue_model():
    if TRUEVALUE_CACHE.exists():
        import joblib
        return joblib.load(TRUEVALUE_CACHE)
    return fit_truevalue_model()


_TV_TAIL_CACHE = {}


def _tv_tail(model, age, q50, q90):
    """(knee_booker, value_at_knee, $/BOOKER slope) for the uncapped linear tail at a
    given age, measured from the dense sub-max region."""
    key = (id(model), round(float(age), 2))
    if key in _TV_TAIL_CACHE:
        return _TV_TAIL_CACHE[key]
    lo = float(np.exp(model.predict(np.array([[q50, age]]))[0]))
    hi = float(np.exp(model.predict(np.array([[q90, age]]))[0]))
    slope = max((hi - lo) / max(q90 - q50, 0.5), 0.0)
    out = (q90, hi, slope)
    _TV_TAIL_CACHE[key] = out
    return out


def truevalue_predict(booker_score, age, blob=None):
    """Predicted AAV at a given (BOOKER, age) on the monotonic surface, with an
    uncapped linear tail above the dense-data region (value can exceed the supermax)."""
    blob = blob or _load_truevalue_model()
    b = float(booker_score) if booker_score is not None and not (
        isinstance(booker_score, float) and np.isnan(booker_score)) else 0.0
    if blob.get("kind") == "linear":
        return float(blob["b0"] + blob["b1"] * b)
    q50, q90 = blob.get("booker_q", (0.5, 2.1))
    if b <= q90:
        pred = blob["model"].predict(np.array([[b, float(age)]]))[0]
        return float(np.exp(pred)) if blob.get("log") else float(pred)
    knee, val_knee, slope = _tv_tail(blob["model"], age, q50, q90)
    return float(val_knee + slope * (b - knee))


def predict_true_value(booker_score):
    """True Value = predicted AAV at the young-prime reference age (age penalty
    removed), so value rises monotonically with BOOKER skill alone."""
    blob = _load_truevalue_model()
    return truevalue_predict(booker_score, blob["ref_age"], blob)


def lookup_known_aav_2026(name):
    fa = load_fa_signings()
    nm = norm_name(name)
    hit = fa[fa.nm == nm]
    if not hit.empty:
        r = hit.sort_values("sign_year", ascending=False).iloc[0]
        return float(r.aav_2026), int(r.yrs)

    sal = load_salary_history()
    hit = sal[(sal.nm == nm) & (sal.season == TARGET_YEAR)]
    if not hit.empty:
        r = hit.sort_values("salary", ascending=False).iloc[0]
        return float(inflate_salary(r.salary, TARGET_YEAR)), None

    hit = sal[sal.nm == nm].sort_values("season", ascending=False)
    if not hit.empty:
        r = hit.iloc[0]
        return float(inflate_salary(r.salary, int(r.season))), None
    return None, None


def player_contract_row(name, pos, age, booker_score, years_pro=4, waa_by_name=None):
    """True Value (skill-based fair AAV, age penalty removed) vs the actual deal.

    `booker_score` is the player's BOOKER score (predictive WAA / 3000 poss)."""
    tv = predict_true_value(booker_score)
    actual, yrs = lookup_known_aav_2026(name)
    surplus = int(round(tv - actual)) if actual is not None else None
    return {
        "trueValue": int(round(tv)),
        "fairAav2026": int(round(tv)),   # back-compat alias (now holds True Value)
        "marketAav2026": int(round(actual)) if actual is not None else None,
        "surplus": surplus,
        "contractYears": yrs,
        "isKnownDeal": actual is not None,
    }


def cap_rules_payload():
    return {
        "targetYear": TARGET_YEAR,
        "cap": int(cap_for_season(TARGET_YEAR)),
        "tax": int(TAX_LINE_2026),
        "apron1": int(APR1_2026),
        "apron2": int(APR2_2026),
        "mle": int(MLE_2026),
        "taxpayerMle": int(TAXPAYER_MLE_2026),
        "vetMin": {str(k): int(v) for k, v in VET_MIN_2026.items()},
    }


def build_waa_name_map(data, season):
    by_ss = build_waa_by_season()
    waa = {nm: v for (nm, ss), v in by_ss.items() if ss == season}
    if not waa and data is not None:
        for (nm, ss), v in by_ss.items():
            if ss <= season:
                waa[nm] = v
    return waa


def inflation_table():
    seasons = list(range(2000, 2027))
    cap_tgt = cap_for_season(TARGET_YEAR)
    return [{
        "season": s,
        "cap": int(cap_for_season(s)),
        "factorTo2026": round(cap_tgt / cap_for_season(s), 4),
    } for s in seasons]


def attach_contracts(players, waa_by_name, ages, years_pro, default_pos="SF"):
    out = []
    for p in players:
        nm = norm_name(p.get("player", ""))
        age = ages.get(nm, 27.0)
        pos = p.get("pos", default_pos)
        waa = waa_by_name.get(nm, p.get("waaTotal", 0.0))
        yp = years_pro.get(nm, 4)
        c = player_contract_row(p.get("player", nm), pos, age, waa, yp, waa_by_name)
        row = dict(p)
        row.update(c)
        row["age"] = round(age, 1)
        row["yearsPro"] = yp
        out.append(row)
    return out
