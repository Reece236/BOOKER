"""Export model outputs to a single JS data file the static dashboard loads."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
DASH = HERE / "dashboard" / "data"
DASH.mkdir(parents=True, exist_ok=True)


def _read(path):
    return pd.read_csv(path) if path.exists() else None


def load_preseason():
    df = _read(CACHE / "preseason_all.csv")
    if df is None:
        return []
    return [{
        "season": int(r.season), "team": r.team,
        "predNet": float(r.pred_net), "projWins": float(r.proj_wins),
        "simMean": float(r.sim_mean), "simSd": float(r.sim_sd),
        "p10": int(r.p10), "p50": int(r.p50), "p90": int(r.p90),
        "pPlayoff": float(r.p_playoff),
        "actualWins": (None if pd.isna(r.actual_wins) else int(r.actual_wins)),
    } for r in df.itertuples()]


def load_timeline():
    df = _read(CACHE / "inseason_timeline_all.csv")
    if df is None:
        return []
    return [{
        "season": int(r.season), "date": r.date, "frac": float(r.frac),
        "team": r.team, "gp": int(r.games_played), "wtd": int(r.wins_to_date),
        "projFinal": float(r.proj_final), "predNet": float(r.pred_net),
        "actualWins": (None if pd.isna(r.actual_wins) else int(r.actual_wins)),
    } for r in df.itertuples()]


def load_game_metrics():
    df = _read(CACHE / "game_metrics.csv")
    if df is None:
        return []
    out = []
    for r in df.itertuples():
        season = str(r.season)
        label = "Pooled" if season == "POOLED" else f"{int(float(season))-1}-{season[2:4]}"
        row = {"season": (None if season == "POOLED" else int(float(season))),
               "label": label, "games": int(r.games),
               "modelLogloss": _f(r, "model_logloss"), "modelBrier": _f(r, "model_brier"),
               "modelAcc": _f(r, "model_acc"), "marketLogloss": _f(r, "market_logloss"),
               "marketBrier": _f(r, "market_brier"), "marketAcc": _f(r, "market_acc"),
               "marketGames": _i(r, "market_games"),
               "roi": _f(r, "roi"), "nBets": _i(r, "n_bets")}
        out.append(row)
    return out


def load_calibration():
    df = _read(CACHE / "game_calibration.csv")
    if df is None:
        return []
    return [{"binMid": float(r.bin_mid), "predMean": float(r.pred_mean),
             "empirical": float(r.empirical), "count": int(r.count)}
            for r in df.itertuples()]


def load_recent_games():
    df = _read(CACHE / "game_predictions_all.csv")
    if df is None:
        return []
    latest = int(df.season.max())
    g = df[df.season == latest].copy()
    return [{
        "season": int(r.season), "date": r.date, "home": r.home, "away": r.away,
        "modelPHome": round(float(r.model_p_home), 3),
        "marketPHome": (None if pd.isna(r.market_p_home) else round(float(r.market_p_home), 3)),
        "homeWin": int(r.home_win),
    } for r in g.itertuples()]


def _f(r, k):
    v = getattr(r, k, None)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)


def _i(r, k):
    v = getattr(r, k, None)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else int(v)

TEAM_NAME = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "CHI": "Chicago Bulls", "CHO": "Charlotte Hornets", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def load_enhanced_players():
    bay = HERE / "booker_bayesian_ratings.csv"
    if bay.exists():
        df = pd.read_csv(bay)
        out = {}
        for r in df.itertuples():
            key = (int(r.PLAYER_ID), int(r.season))
            out[key] = {
                "waaOff": round(float(r.waa_off), 2),
                "waaDef": round(float(r.waa_def), 2),
                "waaOff100": round(float(r.impact_off), 2),
                "waaDef100": round(float(r.impact_def), 2),
                "rankOff": int(r.rank) if hasattr(r, "rank") else None,
                "rankDef": int(r.rank),
                "sdOff": round(float(r.sd_off), 2),
                "sdDef": round(float(r.sd_def), 2),
            }
        return out
    path = HERE / "booker_waa_enhanced_ratings.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for r in df.itertuples():
        key = (int(r.pid), int(r.season))
        out[key] = {
            "waaOff": round(float(r.waa_off), 2),
            "waaDef": round(float(r.waa_def), 2),
            "waaOff100": round(float(r.impact_off), 2),
            "waaDef100": round(float(r.impact_def), 2),
            "rankOff": int(r.rank_off),
            "rankDef": int(r.rank_def),
        }
    return out


def load_trade():
    try:
        from forecast import player_impacts as pi
        from forecast import minutes_model as mm
        from forecast.trade_sim import export_trade_payload
        data = pi.BookerData(seasons=range(2015, 2028))
        season = 2027 if 2027 in data.GAMES else max(data.seasons)
        payload = export_trade_payload(data, season=season)
        train = pi.prior_train_seasons(data, season)
        k, c = pi.fit_net_to_wins(data, train)
        from forecast import contract_value as cv
        from forecast import enhanced_impacts as ei
        enh = ei.build_enhanced(data, train, season)
        # Roll the roster up over *projected healthy* rotation minutes (fixed
        # 240-min/game team budget, replacement-level tail) rather than cloned
        # injury-shortened minutes -- this is what feeds the dashboard win totals
        # and the Lineup Lab base rotation.
        proj_min = mm.project_minutes(data, season)
        budget = pi.TEAM_BUDGET
        _, _, net_tid = ei.aggregate_off_def(
            data, enh, season, target_season=season, minutes=proj_min, budget=budget)
        waa_map = cv.build_waa_name_map(data, season)
        cv.fit_model(waa_map)
        ages = {}
        age_map, latest = cv._player_ages()
        for (nm, ss), ag in age_map.items():
            if ss == season:
                ages[nm] = ag
        for nm in latest.index:
            ages.setdefault(nm, float(latest.loc[nm, "age"]))
        yos = {}
        sal = cv.load_salary_history()
        for nm, n in sal.groupby("nm").season.nunique().items():
            yos[nm] = int(n)
        fa = cv.load_fa_signings()
        pos_by_nm = fa.sort_values("sign_year").groupby("nm").POS.last().to_dict()
        bk = pd.read_csv(pi.PLAYER_DATA)
        bk["nm"] = bk.playerName.map(cv.norm_name)
        pos_bk = bk[bk.season == season].groupby("nm").position.last().to_dict()
        for nm, p in pos_bk.items():
            pos_by_nm.setdefault(nm, p)
        abbr = dict(zip(data.TEAMS[season].TEAM_ID, data.TEAMS[season].ABBR))
        team_net = {abbr[t]: round(float(v), 2) for t, v in net_tid.items() if t in abbr}
        team_wins = {t: round(k * v + c, 1) for t, v in team_net.items()}
        pl = data.PLAYERS[season]
        # observed team minutes -- kept only for the BOOKER per-minute rate, which
        # pairs observed WAA with observed presence
        obs_team_min = {}
        for pid, tid, mn in zip(pl.PLAYER_ID, pl.TEAM_ID, pl.MINUTES):
            ab = abbr.get(tid)
            if ab:
                obs_team_min[ab] = obs_team_min.get(ab, 0.0) + float(mn)
        # projected rotation minutes per team (~19,680; feeds teamMinutes + Lineup Lab)
        team_min = {}
        for pid, tid in zip(pl.PLAYER_ID, pl.TEAM_ID):
            ab = abbr.get(tid)
            if ab:
                team_min[ab] = team_min.get(ab, 0.0) + proj_min.get(int(pid), 0.0)
        players = []
        for r in payload["components"]:
            ab = r["team"]
            tm = obs_team_min.get(ab, 1.0)
            pres = r["minutes"] / (tm / 5.0) if tm > 0 else 0
            pmin = proj_min.get(int(r["pid"]), 0.0)
            proj_pres = pmin / (budget / 5.0)
            nm = cv.norm_name(r["player"])
            pos = pos_by_nm.get(nm, "SF")
            age = ages.get(nm, 27.0)
            yp = yos.get(nm, 4)
            # BOOKER score (per-3000-poss rate) from this roster's WAA + presence
            booker = (r["waa_total"] / pres * (3000.0 / 8200.0)) if pres > 0 else 0.0
            contract = cv.player_contract_row(
                r["player"], pos, age, booker, yp, waa_map)
            players.append({
                "pid": r["pid"], "player": r["player"], "team": ab,
                "minutes": int(round(pmin)),
                "projMin": int(round(pmin)),
                "impactTotal": r["impact_total"],
                "impactOff": r["impact_off"], "impactDef": r["impact_def"],
                "netContrib": round(r["impact_total"] * proj_pres, 2),
                "waaOff": r["waa_off"], "waaDef": r["waa_def"], "waaTotal": r["waa_total"],
                "pos": pos, "age": round(age, 1), "yearsPro": yp,
                **contract,
            })
        pre = _read(CACHE / "preseason_all.csv")
        sim_wins = {}
        if pre is not None:
            ps = pre[pre.season == season]
            sim_wins = {r.team: float(r.sim_mean) for r in ps.itertuples()}
        return {
            "season": season,
            "k": round(k, 3), "c": round(c, 1),
            "teamBudget": int(budget),
            "replacementImpact": pi.REPLACEMENT_IMPACT,
            "teamNet": team_net, "teamWins": team_wins, "teamSimWins": sim_wins,
            "teamMinutes": {k: int(v) for k, v in team_min.items()},
            "players": players,
            "maxAssets": 6,
            "capRules": cv.cap_rules_payload(),
            "inflation": cv.inflation_table(),
        }
    except Exception as exc:
        print(f"trade payload skipped: {exc}")
        return None


def _contract_lookups():
    from forecast import contract_value as cv
    from forecast import player_impacts as pi

    fa = cv.load_fa_signings()
    pos_by_nm = fa.sort_values("sign_year").groupby("nm").POS.last().to_dict()
    bk = pd.read_csv(pi.PLAYER_DATA)
    bk["nm"] = bk.playerName.map(cv.norm_name)
    pos_bk = bk.groupby("nm").position.last().to_dict()
    for nm, p in pos_bk.items():
        pos_by_nm.setdefault(nm, p)
    age_map, latest = cv._player_ages()
    latest_age = {nm: float(latest.loc[nm, "age"]) for nm in latest.index}
    yos = {nm: int(n) for nm, n in cv.load_salary_history().groupby("nm").season.nunique().items()}
    return pos_by_nm, age_map, latest_age, yos


MASTER = HERE.parent / "nba_master_dataset_with_archetypes.csv"

# savant/databallr-style skill set: (key, label, source, column, higher_is_better)
#   source "box"   -> per-32 / efficiency column from the master dataset
#   source "model" -> a field already on the player row (BookerFormer impact / BOOKER)
SKILL_DEFS = [
    ("offense", "Offense", "model", "bfOff100", True),
    ("defense", "Defense", "model", "bfDef100", True),
    ("scoring", "Scoring", "box", "points/32", True),
    ("efficiency", "Efficiency", "box", "tsPercent", True),
    ("playmaking", "Playmaking", "box", "assists/32", True),
    ("rebounding", "Rebounding", "box", "totalRb/32", True),
    ("steals", "Steals", "box", "steals/32", True),
    ("rim_protect", "Rim Protection", "box", "blocks/32", True),
    ("three_volume", "3PT Volume", "box", "3p_fg_attempted/32", True),
    ("three_pct", "3PT %", "box", "three_pct", True),
    ("usage", "Usage", "box", "usagePercent", True),
    # shot-quality (joins on player-id + season; from cache/shot_quality.csv)
    ("shot_making", "Shot-Making", "shotq", "pts_oe100", True),   # pts over expected/100 shots
    ("shot_difficulty", "Shot Difficulty", "shotq", "xfg_inv", True),  # harder diet = higher
    ("self_creation", "Self-Creation", "shotq", "self_create", True),  # off-the-dribble share
    # shot-defense (cache/shot_defense.csv): rim-attempt deterrence + make-limiting
    ("rim_deterrence", "Rim Deterrence", "shotd", "rim_deter", True),
    ("make_limiting", "Make Limiting", "shotd", "suppression", True),
]
# stat-leaders categories shown on the Stat Leaders tab (subset of SKILL_DEFS keys)
LEADER_STATS = ["scoring", "efficiency", "three_pct", "three_volume", "playmaking",
                "rebounding", "steals", "rim_protect", "usage", "offense", "defense"]
SKILL_MIN_MINUTES = 500


def _percentiles(values):
    """Map each value to a 0-100 percentile rank (ties share the average rank)."""
    import numpy as np
    arr = np.array(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty(len(arr))
    ranks[order] = np.arange(len(arr))
    # average-rank for ties
    out = {}
    for v in np.unique(arr):
        mask = arr == v
        out[v] = ranks[mask].mean()
    pct = np.array([out[v] for v in arr])
    return (100.0 * pct / max(len(arr) - 1, 1))


def build_skill_profiles(players):
    """Attach a per-player-season `skills` dict of {label, pct, val} for the player
    page breakdown. Box skills come from the master dataset (per-32 / efficiency),
    impact skills from the BookerFormer rating already on the row. Percentiles are
    computed within each season among players with >= SKILL_MIN_MINUTES minutes."""
    # master dataset uses Basketball-Reference string ids, so join on (name, season)
    from forecast import player_impacts as pi
    box, archetype = {}, {}
    if MASTER.exists():
        m = pd.read_csv(MASTER)
        # derived 3PT% (require ~1 attempt/game so the rate isn't noise)
        att = pd.to_numeric(m.get("total_threeAttempts"), errors="coerce")
        made = pd.to_numeric(m.get("total_threeFg"), errors="coerce")
        m["three_pct"] = np.where(att >= 82, made / att.replace(0, np.nan), np.nan)
        needed = [c for (_, _, src, c, _) in SKILL_DEFS if src == "box"]
        m["_key"] = list(zip(m.playerName.map(pi.norm_name),
                             pd.to_numeric(m.season, errors="coerce").astype("Int64")))
        # one row per (name, season): keep the largest-minutes stint (traded players
        # appear once per team), then index uniquely
        m = m.sort_values("minutesPlayed").drop_duplicates("_key", keep="last")
        box = m.set_index("_key")[needed].to_dict("index")   # {(name,season): {col: val}}
        if "archetype" in m.columns:
            archetype = m.set_index("_key")["archetype"].to_dict()

    # shot-quality metrics, keyed on (PLAYER_ID, season)
    shotq = {}
    sqpath = CACHE / "shot_quality.csv"
    if sqpath.exists():
        sq = pd.read_csv(sqpath)
        sq["xfg_inv"] = (1.0 - pd.to_numeric(sq.xfg, errors="coerce")).round(4)
        for r in sq.itertuples():
            shotq[(int(r.PLAYER_ID), int(r.season))] = {
                "pts_oe100": getattr(r, "pts_oe100", None),
                "xfg_inv": getattr(r, "xfg_inv", None),
                "self_create": getattr(r, "self_create", None),
            }
    # shot-defense metrics, keyed on (PLAYER_ID, season)
    shotd = {}
    sdpath = CACHE / "shot_defense.csv"
    if sdpath.exists():
        sdf = pd.read_csv(sdpath)
        for r in sdf.itertuples():
            shotd[(int(r.PLAYER_ID), int(r.season))] = {
                "rim_deter": getattr(r, "rim_deter", None),
                "suppression": getattr(r, "suppression", None),
            }

    def _bkey(p):
        return (pi.norm_name(p["player"]), p["season"])

    by_season = {}
    for p in players:
        by_season.setdefault(p["season"], []).append(p)

    for season, rows in by_season.items():
        pool = [p for p in rows if p.get("min", 0) >= SKILL_MIN_MINUTES]
        if len(pool) < 5:
            pool = rows
        for key, label, source, col, _ in SKILL_DEFS:
            vals, idx = [], []
            for i, p in enumerate(pool):
                if source == "model":
                    v = p.get(col)
                elif source == "shotq":
                    v = shotq.get((p["pid"], p["season"]), {}).get(col)
                elif source == "shotd":
                    v = shotd.get((p["pid"], p["season"]), {}).get(col)
                else:
                    b = box.get(_bkey(p))
                    v = b.get(col) if b is not None else None
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                vals.append(float(v)); idx.append(i)
            if not vals:
                continue
            pcts = _percentiles(vals)
            for j, i in enumerate(idx):
                pool[i].setdefault("skills", {})[key] = {
                    "label": label, "pct": round(float(pcts[j])), "val": round(vals[j], 2),
                }
        for p in pool:
            a = archetype.get(_bkey(p))
            if a is not None and not (isinstance(a, float) and np.isnan(a)):
                p["archetype"] = str(a)
    return players


def build_diagnostics(players):
    """Bayesian-model diagnostics block: uncertainty-vs-minutes, True Value fit +
    FA scatter, and the out-of-sample backtest summary."""
    from forecast import contract_value as cv

    # latest season that actually has posterior SD (projection seasons have none)
    sd_seasons = [p["season"] for p in players if p.get("sdOff") is not None]
    latest = max(sd_seasons) if sd_seasons else max(p["season"] for p in players)
    rated = [p for p in players if p.get("sdOff") is not None and p["season"] == latest]

    # uncertainty narrows with minutes: bin players by minutes, mean sd
    bins = [(250, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
    sd_by_min = []
    for lo, hi in bins:
        grp = [p for p in rated if lo <= p["min"] < hi]
        if grp:
            sd_by_min.append({
                "bin": f"{lo}-{hi}", "n": len(grp),
                "sdOff": round(float(np.mean([p["sdOff"] for p in grp])), 3),
                "sdDef": round(float(np.mean([p["sdDef"] for p in grp])), 3),
            })

    # True Value model + FA scatter (BOOKER prior season vs signed AAV)
    tvblob = cv._load_truevalue_model()
    booker = cv.build_booker_by_season()
    fa = cv.load_fa_signings()
    fa_scatter = []
    for r in fa.itertuples():
        b = booker.get((r.nm, int(r.sign_year)))
        if b is None or r.aav_2026 is None or np.isnan(r.aav_2026) or r.aav_2026 < 500_000:
            continue
        fa_scatter.append({"player": r.player, "booker": round(float(b), 2),
                           "aav": int(round(float(r.aav_2026)))})
    # fitted monotonic surface: True Value curve (young ref age) + an older-age curve
    # to visualize the age penalty
    bgrid = [round(0.5 * i, 1) for i in range(0, 25)]
    curve_young = [{"booker": b, "aav": int(round(cv.truevalue_predict(b, tvblob["ref_age"], tvblob)))}
                   for b in bgrid]
    curve_old = [{"booker": b, "aav": int(round(cv.truevalue_predict(b, 33.0, tvblob)))}
                 for b in bgrid]

    # credible intervals for the top players (latest season)
    top = sorted(rated, key=lambda p: p.get("bookerScore", -99), reverse=True)[:20]
    intervals = [{
        "player": p["player"], "team": p["team"], "booker": p.get("bookerScore"),
        "off": p.get("bfOff100"), "offSd": p.get("sdOff"),
        "def": p.get("bfDef100"), "defSd": p.get("sdDef"),
    } for p in top]

    return {
        "model": "BookerFormer (variational Bayesian RAPM)",
        "sdByMinutes": sd_by_min,
        "trueValue": {"kind": tvblob.get("kind", "gbm"), "n": tvblob["n"],
                      "refAge": tvblob["ref_age"], "curveYoung": curve_young,
                      "curveOld": curve_old},
        "faScatter": fa_scatter,
        "intervals": intervals,
        # out-of-sample backtest (bookerformer_backtest.py, targets 2024-25)
        "backtest": {
            "rows": [
                {"model": "Ridge RAPM (prior)", "netRmse": 4.04, "winsR2": 0.553, "cov90": None},
                {"model": "BookerFormer (additive)", "netRmse": 4.00, "winsR2": 0.608, "cov90": 0.893},
                {"model": "BookerFormer + attention", "netRmse": 4.58, "winsR2": 0.400, "cov90": 0.893},
            ],
            "note": "Trained on prior 3 seasons, evaluated out-of-sample. Additive "
                    "Bayesian model beats ridge on team net/wins with ~90% interval "
                    "coverage; the transformer layer is opt-in (did not improve OOS).",
        },
    }


def main():
    from forecast import contract_value as cv
    from forecast import leaderboard_data as lb
    from forecast import player_impacts as pi

    ratings = pd.read_csv(HERE / "booker_waa_ratings_by_year.csv")
    ratings = ratings.rename(columns={"PLAYER_ID": "pid"})
    data = pi.BookerData(seasons=range(2015, 2028))
    model_map = lb.load_model_waa_map(data)
    proj_2027 = lb.build_2027_projections(data)
    pos_by_nm, age_map, latest_age, yos = _contract_lookups()
    waa_ss = cv.build_waa_by_season()
    if not cv.MODEL_CACHE.exists():
        cv.fit_model(cv.build_waa_name_map(data, 2026))
    tv = cv.fit_truevalue_model()   # monotonic GBM: AAV ~ f(BOOKER+, age-)
    print(f"True Value model: monotonic {tv.get('kind', 'gbm')} (BOOKER+, age-), "
          f"n={tv['n']} FA signings, value read at age {tv['ref_age']}")

    players = []
    for r in ratings.itertuples():
        key = (int(r.pid), int(r.season))
        row = {
            "pid": int(r.pid), "season": int(r.season), "rank": int(r.rank),
            "player": r.player, "team": r.team,
            "min": round(float(r.minutes)), "prior": round(float(r.prior), 2),
            "waa100": round(float(r.impact_per100), 2),
            "waa": round(float(r.waa_wins), 2),
            "waaLegacy": round(float(r.waa_wins), 2),
        }
        extra = model_map.get(key)
        if extra:
            row.update({
                "waaOff": extra["waaOff"],
                "waaDef": extra["waaDef"],
                "waaModel": extra["waaModel"],
                "waaOff100": extra["waaOff100"],
                "waaDef100": extra["waaDef100"],
                "waaModel100": extra["waaModel100"],
                "modelType": extra["modelType"],
                "waa": extra["waaModel"],
                "waa100": extra["waaModel100"],
            })
            # enhanced (teammate-fit ridge) numbers, kept as a leaderboard comparison
            if extra.get("waaEnhanced") is not None:
                row["waaEnhanced"] = extra["waaEnhanced"]
                row["waaEnhanced100"] = extra.get("waaEnhanced100")
            if extra.get("waa32") is not None:
                row["waa32"] = extra["waa32"]
                row["waa32Off"] = extra.get("waa32Off")
                row["waa32Def"] = extra.get("waa32Def")
                row["rankWaa32"] = extra.get("rankWaa32")
            if extra.get("sdOff") is not None:
                row["sdOff"] = extra["sdOff"]
                row["sdDef"] = extra["sdDef"]
            # BookerFormer Bayesian overlay: O/D rating per-100 + label, paired with
            # sdOff/sdDef above so the UI can show a calibrated credible interval.
            if extra.get("uncModel") is not None:
                row["uncModel"] = extra["uncModel"]
                row["bfOff100"] = extra.get("waaBayesianOff100")
                row["bfDef100"] = extra.get("waaBayesianDef100")
                row["bfTot100"] = extra.get("waaBayesian100")
            # BOOKER score: predictive WAA / 3000 poss (skill, no aging)
            if extra.get("bookerScore") is not None:
                row["bookerScore"] = extra["bookerScore"]
                row["bookerOff"] = extra.get("bookerOff")
                row["bookerDef"] = extra.get("bookerDef")
        proj = proj_2027.get(int(r.pid))
        if proj:
            row.update(proj)
        nm = cv.norm_name(r.player)
        # True Value = skill-based fair AAV (BOOKER score, age penalty removed).
        # Pre-2018 rows have no BOOKER score -> derive one from waaModel & minutes
        # (BOOKER = WAA scaled to a 3000-poss / ~1440-min workload).
        bscore = row.get("bookerScore")
        if bscore is None and row["min"] > 0:
            bscore = round(row.get("waaModel", row["waa"]) * 1440.0 / row["min"], 2)
        age = age_map.get((nm, int(r.season)), latest_age.get(nm, 27.0))
        lb.attach_contract_fields(row, pos_by_nm, {nm: age}, yos, bscore)
        players.append(row)

    # ---- predictive 2026-27 rows (projection season: no box stats, forecasts only)
    PROJ = 2027
    if PROJ in data.PLAYERS and proj_2027:
        abbr = dict(zip(data.TEAMS[PROJ].TEAM_ID, data.TEAMS[PROJ].ABBR)) \
            if PROJ in data.TEAMS else {}
        for prow in data.PLAYERS[PROJ].itertuples():
            pid = int(prow.PLAYER_ID)
            pj = proj_2027.get(pid)
            mins = float(prow.MINUTES)
            if pj is None or mins < 250:
                continue
            waa = pj["waaProj2027"]
            row = {
                "pid": pid, "season": PROJ, "player": prow.NAME,
                "team": abbr.get(prow.TEAM_ID, "?"), "min": round(mins),
                "waa": waa, "waaModel": waa,
                "waaOff": pj["waaOffProj2027"], "waaDef": pj["waaDefProj2027"],
                "bookerScore": round(waa * 1440.0 / mins, 2) if mins > 0 else None,
                "bookerOff": round(pj["waaOffProj2027"] * 1440.0 / mins, 2) if mins > 0 else None,
                "bookerDef": round(pj["waaDefProj2027"] * 1440.0 / mins, 2) if mins > 0 else None,
                "grade2027": pj["grade2027"], "projRank2027": pj["projRank2027"],
                "modelType": "projection", "predictive": True,
            }
            nm = cv.norm_name(prow.NAME)
            age = age_map.get((nm, PROJ), latest_age.get(nm, 27.0))
            lb.attach_contract_fields(row, pos_by_nm, {nm: age}, yos, row["bookerScore"])
            players.append(row)

    by_season = {}
    for row in players:
        by_season.setdefault(row["season"], []).append(row)
    for season, rows in by_season.items():
        rows.sort(key=lambda x: x.get("waaModel", x["waa"]), reverse=True)
        for i, row in enumerate(rows, 1):
            row["rankModel"] = i

    build_skill_profiles(players)            # savant-style percentile bars
    diagnostics = build_diagnostics(players)  # Bayesian calibration + True Value fit

    bt = pd.read_csv(HERE / "waa_backtest_team_predictions.csv")
    team_pred = [{
        "season": int(r.season), "team": r.team,
        "actualNet": float(r.actual_net), "predNet": float(r.pred_net),
        "actualWins": int(r.actual_wins), "predWins": float(r.pred_wins),
        "oldWins": (None if pd.isna(r.old_model_wins) else float(r.old_model_wins)),
        "winErr": round(float(r.pred_wins) - float(r.actual_wins), 1),
    } for r in bt.itertuples()]

    fm = pd.read_csv(HERE / "waa_backtest_metrics.csv")
    metrics = [{
        "season": (None if str(r.season) == "POOLED" else int(r.season)),
        "label": ("Pooled" if str(r.season) == "POOLED" else f"{int(r.season)-1}-{str(int(r.season))[2:]}"),
        "netRmse": float(r.net_rmse), "netR2": float(r.net_r2),
        "winsRmse": float(r.wins_rmse), "winsMae": float(r.wins_mae),
        "winsR2": float(r.wins_r2), "winsSlope": float(r.wins_slope),
    } for r in fm.itertuples()]

    preseason = load_preseason()
    timeline = load_timeline()
    game_metrics = load_game_metrics()
    calibration = load_calibration()
    recent_games = load_recent_games()
    forecast_seasons = sorted({p["season"] for p in preseason})
    seasons = sorted({p["season"] for p in players} | set(forecast_seasons))
    trade = load_trade()
    payload = {
        "players": players,
        "teamPred": team_pred,
        "metrics": metrics,
        "seasons": seasons,
        "teamNames": TEAM_NAME,
        "baselines": {"predict41": 11.99, "oldBox": 9.01},
        "preseason": preseason,
        "timeline": timeline,
        "gameMetrics": game_metrics,
        "calibration": calibration,
        "recentGames": recent_games,
        "forecastSeasons": forecast_seasons,
        "trade": trade,
        "diagnostics": diagnostics,
        "generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
    }
    out = DASH / "data.js"
    out.write_text("window.BOOKER = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)  "
          f"players={len(players)} teamPred={len(team_pred)} seasons={seasons[0]}-{seasons[-1]} "
          f"preseason={len(preseason)} timeline={len(timeline)} "
          f"gameMetrics={len(game_metrics)} recentGames={len(recent_games)}")


if __name__ == "__main__":
    main()
