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
        "pChamp": (float(r.p_champ) if hasattr(r, "p_champ") and pd.notna(r.p_champ) else None),
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


def load_games_and_ratings():
    """FotMob-style match data: every 2026 game (result + model/market pregame
    probs) + the full 2027 schedule (pregame probs from shrunk preseason nets),
    plus per-player per-game ratings (forecast/game_ratings.py) mapped to a
    0-10 scale: r = 6.6 + 1.15*asinh(g/6) -- league-average game 6.6, a +7.6
    (p90) game ~7.9, Jokic's best 2026 games ~9.7, clipped [2, 10]."""
    from scipy.stats import norm as _norm
    from forecast import player_impacts as pi
    games, gid_ix = [], {}
    for s in (2025, 2026, 2027):
        gp = _read(CACHE / f"games_{s}.csv")
        if gp is None:
            continue
        preds = {}
        pr = _read(CACHE / f"game_predictions_{s}.csv")
        if pr is not None:
            for r in pr.itertuples():
                preds[(str(r.date), r.home, r.away)] = (
                    round(float(r.model_p_home), 3),
                    None if pd.isna(r.market_p_home) else round(float(r.market_p_home), 3))
        nets = {}
        ps = _read(CACHE / f"preseason_{s}.csv")
        if ps is not None:
            nets = dict(zip(ps.team, ps.pred_net))
        for r in gp.itertuples():
            done = pd.notna(r.HOME_PTS) and r.HOME_PTS == r.HOME_PTS
            mp, mk = preds.get((str(r.DATE), r.HOME, r.AWAY), (None, None))
            if mp is None and r.HOME in nets and r.AWAY in nets and \
                    getattr(r, "SEASON_TYPE", "Regular Season") == "Regular Season":
                mp = round(float(_norm.cdf((nets[r.HOME] - nets[r.AWAY]
                                            + pi.HOME_COURT_ADV) / pi.GAME_MARGIN_SD)), 3)
            row = {"id": int(r.GAME_ID), "season": int(r.SEASON), "date": str(r.DATE),
                   "home": r.HOME, "away": r.AWAY,
                   "type": ("P" if getattr(r, "SEASON_TYPE", "") == "Playoffs" else "R"),
                   "pHome": mp, "mHome": mk}
            if done:
                row["hPts"] = int(r.HOME_PTS); row["aPts"] = int(r.AWAY_PTS)
            gid_ix[int(r.GAME_ID)] = len(games)
            games.append(row)
    pgr = _read(CACHE / "player_game_ratings.csv")
    player_games = {}
    if pgr is not None:
        r10 = (6.6 + 1.15 * np.arcsinh(pgr.g / 6.0)).clip(2.0, 10.0).round(1)
        for pid, gid, v in zip(pgr.PLAYER_ID, pgr.GAME_ID, r10):
            gi = gid_ix.get(int(gid))
            if gi is not None:
                player_games.setdefault(int(pid), []).append([gi, float(v)])
        for v in player_games.values():
            v.sort(key=lambda x: x[0])
    return games, player_games


def load_playoff_split():
    """Career playoff-vs-regular split per player from the per-game LOO ratings
    (2018-2026, playoff stints backfilled). {pid: [reg10, po10, poGames, diff100]}.
    Research note (2026-07-20): the split is DESCRIPTIVE -- across 893
    player-postseasons no style feature predicts the playoff differential
    (all FDR q>0.96; Jokic -0.5 on a +12.5 base, Gobert +0.7 = the discourse
    is backwards), so no rating adjustment is made from it."""
    p = CACHE / "player_game_ratings.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d["po"] = d.GAME_ID.astype(str).str.startswith("4")
    d["wg"] = d.g * d.poss
    out = {}
    r10 = lambda g: round(float(np.clip(6.6 + 1.15 * np.arcsinh(g / 6.0), 2.0, 10.0)), 2)
    for pid, g in d.groupby("PLAYER_ID"):
        po = g[g.po]
        if len(po) < 10:
            continue
        reg = g[~g.po]
        if not len(reg):
            continue
        gr = reg.wg.sum() / reg.poss.sum()
        gp = po.wg.sum() / po.poss.sum()
        out[int(pid)] = [r10(gr), r10(gp), int(len(po)), round(float(gp - gr), 2)]
    return out


def load_pred_bracket():
    """Predicted playoff bracket for the upcoming season: seed each conference
    by projected wins, advance the per-series favorite. Series probs use the
    validated machinery: shrunk preseason nets + top-7 rotation blend +
    SERIES_HCA, integrated over the SERIES_SD matchup shock (Gauss-Hermite)."""
    from scipy.stats import norm as _norm
    from forecast import player_impacts as pi
    from forecast import preseason as ps
    pre = _read(CACHE / "preseason_all.csv")
    if pre is None:
        return None
    season = int(pre.season.max())
    d = pre[pre.season == season]
    if d.actual_wins.notna().any():
        return None                               # season already played
    adj = ps.top7_adjust(season) if hasattr(ps, "top7_adjust") else {}
    nets = {r.team: r.pred_net + ps.PLAYOFF_TOP7_W * adj.get(r.team, 0.0)
            for r in d.itertuples()}
    wins = dict(zip(d.team, d.proj_wins))
    x, w = np.polynomial.hermite_e.hermegauss(21)
    def p_series(hi, lo):
        gap = nets[hi] - nets[lo] + ps.SERIES_HCA
        pg = _norm.cdf((gap + x * ps.SERIES_SD) / pi.GAME_MARGIN_SD)
        q = 1 - pg
        ps7 = pg ** 4 * (1 + 4 * q + 10 * q * q + 20 * q ** 3)
        return float((ps7 * w).sum() / w.sum())
    rounds = []
    finalists = []
    for conf in ("E", "W"):
        teams = [t for t in nets if pi.CONFERENCE.get(t, "E") == conf]
        seeds = sorted(teams, key=lambda t: -wins[t])[:8]
        cur = seeds
        pair_order = [(0, 7), (3, 4), (2, 5), (1, 6)]
        rnd_teams = [(cur[i], cur[j]) for i, j in pair_order]
        for rnd in (1, 2, 3):
            nxt = []
            for a, b in rnd_teams:
                hi, lo = (a, b) if wins[a] >= wins[b] else (b, a)
                p = p_series(hi, lo)
                rounds.append({"r": rnd, "conf": conf, "hi": hi, "lo": lo, "p": round(p, 3)})
                nxt.append(hi if p >= 0.5 else lo)
            rnd_teams = [(nxt[i], nxt[i + 1]) for i in range(0, len(nxt) - 1, 2)] if len(nxt) > 1 else []
            if len(nxt) == 1:
                finalists.append(nxt[0])
    if len(finalists) == 2:
        a, b = finalists
        hi, lo = (a, b) if wins[a] >= wins[b] else (b, a)
        p = p_series(hi, lo)
        rounds.append({"r": 4, "conf": "F", "hi": hi, "lo": lo, "p": round(p, 3)})
        champ = hi if p >= 0.5 else lo
        pch = d[d.team == champ].p_champ
        return {"season": season, "rounds": rounds, "champ": champ,
                "pChamp": (round(float(pch.iloc[0]), 3) if len(pch) else None)}
    return None


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
            data, enh, season, target_season=season, minutes=proj_min, budget=budget,
            replacement=mm.replacement_for(data, season))
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
        # latest usage-optimum upside per player (BOOKER usage-regression term)
        role_up = {}
        rvp = CACHE / "role_value.csv"
        if rvp.exists():
            for rr in pd.read_csv(rvp).sort_values("season").itertuples():
                role_up[int(rr.PLAYER_ID)] = float(rr.role_upside)
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
        # latest sticky grade per player (for trade skill/timeline breakdown)
        glat = {}
        gpath = CACHE / "player_grades.csv"
        if gpath.exists():
            gdf = pd.read_csv(gpath).sort_values("season")
            for r in gdf.itertuples():
                glat[int(r.PLAYER_ID)] = (r.grade, r.gradeOff, r.gradeDef)
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
            # BOOKER (predictive +/- per 100, usage regressed toward optimum) -- must
            # match build_bookerformer_ratings so contract fits stay on one scale.
            booker = r["impact_total"] + 0.25 * role_up.get(int(r["pid"]), 0.0)
            contract = cv.player_contract_row(
                r["player"], pos, age, booker, yp, waa_map)
            g = glat.get(int(r["pid"]))
            players.append({
                "pid": r["pid"], "player": r["player"], "team": ab,
                "minutes": int(round(pmin)),
                "projMin": int(round(pmin)),
                "impactTotal": r["impact_total"],
                "impactOff": r["impact_off"], "impactDef": r["impact_def"],
                "netContrib": round(r["impact_total"] * proj_pres, 2),
                "waaOff": r["waa_off"], "waaDef": r["waa_def"], "waaTotal": r["waa_total"],
                "pos": pos, "age": round(age, 1), "yearsPro": yp,
                "grade": (None if g is None else int(g[0])),
                "gradeOff": (None if g is None else int(g[1])),
                "gradeDef": (None if g is None else int(g[2])),
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

# TRUE-SKILL leaderboard: every skill is a difficulty/opportunity-adjusted, EB-shrunk
# TALENT estimate -- not a raw tally. Tuple: (key, label, source, col, raw_col, fmt).
#   source: "model" BookerFormer impact (already adjusted) | "shotq" shot_quality.csv
#   (multi-year, difficulty-adjusted shooting true-skill) | "pbp" pbp_skills.csv
#   (EB-shrunk per-36 rates) | "shotd" shot_defense.csv | "box" master dataset.
#   raw_col -> observed stat shown on hover (None if no comparator).
#   fmt: "pct" (value is a 0-1 rate, displayed x100) | "num" (1-decimal number).
SKILL_DEFS = [
    ("offense", "Offense", "model", "bfOff100", None, "num"),
    ("defense", "Defense", "model", "bfDef100", None, "num"),
    ("three_pct", "True 3P%", "shotq", "true_3p", "raw_3p", "pct"),
    ("adj_three", "Adj 3P% (shot diet)", "shotdiff", "d3p_adj", "raw_3p", "pct"),
    ("rim_finish", "True Rim FG%", "shotq", "true_rim", "raw_rim", "pct"),
    ("efficiency", "True eFG%", "shotq", "true_efg", "raw_efg", "pct"),
    ("shot_making", "Shot-Making", "shotq", "shot_making", "pts_oe100", "num"),
    ("self_creation", "Self-Creation", "shotq", "self_create", None, "pct"),
    ("off_gravity", "Off-Ball Gravity", "grav", "off_gravity", None, "num"),
    ("on_gravity", "On-Ball Gravity", "grav", "on_gravity", None, "num"),
    ("playmaking", "Playmaking", "pbp", "true_ast36", "ast36", "num"),
    ("creation", "Shot Creation", "pbp", "true_create36", "create36", "num"),
    ("foul_draw", "Foul Drawing", "pbp", "true_fta36", "fta36", "num"),
    ("free_throw", "Free-Throw %", "pbp", "true_ft_pct", "ft_pct", "pct"),
    ("ball_security", "Ball Security", "pbp", "true_tov_pct", "tov_pct", "pct", -1),
    ("rebounding", "Rebounding", "pbp", "true_reb36", "reb36", "num"),
    ("steals", "Steals", "pbp", "true_stl36", "stl36", "num"),
    ("rim_protect", "Rim Protection", "pbp", "true_blk36", "blk36", "num"),
    ("discipline", "Discipline (low fouls)", "pbp", "true_pf36", "pf36", "num", -1),
    ("rim_contest", "Rim Contest", "shotd", "rim_contest", None, "num"),
    ("perimeter_contest", "Perimeter Contest", "shotd", "perim_contest", None, "num"),
    ("rim_deterrence", "Rim Deterrence", "shotd", "rim_deter", None, "num"),
    ("make_limiting", "Make Limiting", "shotd", "suppression", None, "num"),
    ("shot_difficulty", "Shot Difficulty", "shotq", "xfg_inv", None, "num"),
    ("usage", "Usage", "box", "usagePercent", None, "num"),
]
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
        needed = [c for (_, _, src, c, *_) in SKILL_DEFS if src == "box"]
        m["_key"] = list(zip(m.playerName.map(pi.norm_name),
                             pd.to_numeric(m.season, errors="coerce").astype("Int64")))
        # one row per (name, season): keep the largest-minutes stint (traded players
        # appear once per team), then index uniquely
        m = m.sort_values("minutesPlayed").drop_duplicates("_key", keep="last")
        box = m.set_index("_key")[needed].to_dict("index")   # {(name,season): {col: val}}
        if "archetype" in m.columns:
            archetype = m.set_index("_key")["archetype"].to_dict()

    # shot-quality TRUE-SKILL metrics (multi-year, difficulty-adjusted), keyed (pid, season)
    shotq = {}
    sqpath = CACHE / "shot_quality.csv"
    if sqpath.exists():
        sq = pd.read_csv(sqpath)
        sq["xfg_inv"] = (1.0 - pd.to_numeric(sq.xfg, errors="coerce")).round(4)
        sqcols = ["true_3p", "raw_3p", "true_rim", "raw_rim", "true_efg", "raw_efg",
                  "shot_making", "pts_oe100", "self_create", "xfg_inv"]
        for r in sq.itertuples():
            shotq[(int(r.PLAYER_ID), int(r.season))] = {c: getattr(r, c, None) for c in sqcols}
    # PBP per-36 TRUE-SKILL rate metrics (assists/steals/blocks/rebounds/creation)
    pbp = {}
    pppath = CACHE / "pbp_skills.csv"
    if pppath.exists():
        pp = pd.read_csv(pppath)
        ppcols = ["true_ast36", "ast36", "true_create36", "create36", "true_reb36", "reb36",
                  "true_stl36", "stl36", "true_blk36", "blk36", "true_fta36", "fta36",
                  "true_ft_pct", "ft_pct", "true_tov_pct", "tov_pct", "true_pf36", "pf36"]
        for r in pp.itertuples():
            pbp[(int(r.PLAYER_ID), int(r.season))] = {c: getattr(r, c, None) for c in ppcols}
    # off/on-ball gravity proxies, keyed on (PLAYER_ID, season)
    grav = {}
    gvpath = CACHE / "gravity.csv"
    if gvpath.exists():
        for r in pd.read_csv(gvpath).itertuples():
            grav[(int(r.PLAYER_ID), int(r.season))] = {
                "off_gravity": getattr(r, "off_gravity", None),
                "on_gravity": getattr(r, "on_gravity", None)}
    # defender/diet-adjusted shooting (shot_difficulty.csv): actual vs expected FOR THE
    # SHOT DIET TAKEN (pull-ups/deep/contested), re-anchored to league 3P%. Raw 3P% for
    # the hover comes from shot_quality at the same key.
    shotdiff = {}
    sdfpath = CACHE / "shot_difficulty.csv"
    if sdfpath.exists():
        for r in pd.read_csv(sdfpath).itertuples():
            k = (int(r.PLAYER_ID), int(r.season))
            v = getattr(r, "d3p_adj", None)
            if v is not None and v == v:
                shotdiff[k] = {"d3p_adj": float(v),
                               "raw_3p": (shotq.get(k) or {}).get("raw_3p")}
    # shot-defense metrics (incl. zone-split contest), keyed on (PLAYER_ID, season)
    shotd = {}
    sdpath = CACHE / "shot_defense.csv"
    if sdpath.exists():
        sdf = pd.read_csv(sdpath)
        sdcols = ["rim_deter", "suppression", "rim_contest", "perim_contest"]
        for r in sdf.itertuples():
            shotd[(int(r.PLAYER_ID), int(r.season))] = {c: getattr(r, c, None) for c in sdcols}

    def _bkey(p):
        return (pi.norm_name(p["player"]), p["season"])

    by_season = {}
    for p in players:
        by_season.setdefault(p["season"], []).append(p)

    for season, rows in by_season.items():
        pool = [p for p in rows if p.get("min", 0) >= SKILL_MIN_MINUTES]
        if len(pool) < 5:
            pool = rows
        for key, label, source, col, raw_col, fmt, *rest in SKILL_DEFS:
            sign = rest[0] if rest else 1   # -1 => lower is better (turnovers, fouls)
            vals, raws, idx = [], [], []
            for i, p in enumerate(pool):
                rv = None
                if source == "model":
                    v = p.get(col)
                elif source in ("shotq", "pbp", "grav", "shotdiff"):
                    src = {"shotq": shotq, "pbp": pbp, "grav": grav, "shotdiff": shotdiff}[source]
                    d = src.get((p["pid"], p["season"]), {})
                    v = d.get(col); rv = d.get(raw_col) if raw_col else None
                elif source == "shotd":
                    v = shotd.get((p["pid"], p["season"]), {}).get(col)
                else:
                    b = box.get(_bkey(p))
                    v = b.get(col) if b is not None else None
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                vals.append(float(v)); idx.append(i)
                raws.append(None if rv is None or (isinstance(rv, float) and np.isnan(rv)) else float(rv))
            if not vals:
                continue
            # percentile ranks goodness (lower-is-better skills ranked on negated value)
            pcts = _percentiles([sign * v for v in vals])
            for j, i in enumerate(idx):
                entry = {"label": label, "pct": round(float(pcts[j])),
                         "val": round(vals[j], 3), "fmt": fmt}
                if raws[j] is not None:
                    entry["raw"] = round(raws[j], 3)
                pool[i].setdefault("skills", {})[key] = entry
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
    from forecast import minutes_model as mm
    from forecast import player_impacts as pi

    ratings = pd.read_csv(HERE / "booker_waa_ratings_by_year.csv")
    ratings = ratings.rename(columns={"PLAYER_ID": "pid"})
    data = pi.BookerData(seasons=range(2015, 2028))
    model_map = lb.load_model_waa_map(data)
    proj_2027 = lb.build_2027_projections(data)
    # sticky, age-curved, team-independent grades + 3-yr projections (cache/player_grades.csv)
    grades_map = {}
    gpath = CACHE / "player_grades.csv"
    if gpath.exists():
        gcols = ["grade", "gradeLetter", "gradeOff", "gradeDef", "age",
                 "proj1", "proj1Letter", "proj1Age", "proj2", "proj2Letter", "proj2Age",
                 "proj3", "proj3Letter", "proj3Age",
                 "hybOff", "hybDef", "hybBooker", "hybBkOff", "hybBkDef",
                 "hybWaa", "hybWaaOff", "hybWaaDef"]
        for r in pd.read_csv(gpath).itertuples():
            grades_map[(int(r.PLAYER_ID), int(r.season))] = {
                c: (None if (isinstance(getattr(r, c, None), float) and pd.isna(getattr(r, c)))
                    else getattr(r, c, None)) for c in gcols}
    heights = {}
    hpath = CACHE / "player_heights.csv"
    if hpath.exists():
        for r in pd.read_csv(hpath).itertuples():
            heights[int(r.PLAYER_ID)] = int(r.height_in)
    # lineup-context stats (descriptive: opp/teammate quality + real on-court +/-)
    lctx = {}
    lcpath = CACHE / "lineup_context.csv"
    if lcpath.exists():
        lcc = ["opp_quality", "tm_quality", "real_pm"]
        for r in pd.read_csv(lcpath).itertuples():
            lctx[(int(r.PLAYER_ID), int(r.season))] = {c: getattr(r, c, None) for c in lcc}
    # role/usage value: what a player is worth if used at his optimal usage (skill curve)
    role_map = {}
    rvpath = CACHE / "role_value.csv"
    if rvpath.exists():
        for r in pd.read_csv(rvpath).itertuples():
            role_map[(int(r.PLAYER_ID), int(r.season))] = {
                "usage": float(r.usage), "optUsage": float(r.opt_usage),
                "misuse": float(r.misuse), "tsNow": float(r.ts_now),
                "tsOpt": float(r.ts_opt), "upside": float(r.role_upside)}
    # crunch-time record (career-pooled 2018-2025 from pbp): clutch = last 5 min of a
    # <=5-pt game. Volume (true-shot attempts) + TS in/out of clutch. Descriptive --
    # clutch TS deltas mostly measure the BURDEN a primary option absorbs, so they are
    # surfaced, not priced into BOOKER (lineup-level test: creation preserves clutch
    # offense ~+1.5 ORtg/SD, t=2.0, but macro playoff test = null).
    clutch_map = {}
    clpath = CACHE / "clutch_players.csv"
    if clpath.exists():
        for r in pd.read_csv(clpath, index_col=0).itertuples():
            if r.tsa_c >= 100:
                clutch_map[int(r.Index)] = {
                    "tsaC": int(r.tsa_c), "tsN": round(float(r.ts_n), 3),
                    "tsC": round(float(r.ts_c), 3)}
    # BOOKER-PROJ: next-season player forecast (walk-forward; DARKO-parity overall,
    # best-in-class on team-changers). PLAYER-level only -- the team engine gate
    # rejected it (shrinkage flattens team spread), so trade/lineup sums keep impacts.
    proj_map = {}
    pjpath = CACHE / "booker_proj.csv"
    if pjpath.exists():
        for r in pd.read_csv(pjpath).itertuples():
            # attach to the season the forecast was MADE FROM (proj_for_season - 1)
            proj_map[(int(r.PLAYER_ID), int(r.proj_for_season) - 1)] = float(r.proj_impact)
    # projected next-season minutes (gated ridge + injury blend + overrides; built
    # by data_ingest/rebuild_rosters_2027.py). Lineup Lab seeds player workloads
    # from this instead of a flat 1500.
    projmin_map = {}
    pmp = CACHE / "players_2027.csv"
    if pmp.exists():
        for r in pd.read_csv(pmp).itertuples():
            projmin_map[int(r.PLAYER_ID)] = float(r.MINUTES)
    # ---- play-finisher (lob-center) calibration -------------------------------
    # OOS-validated bias: rim-running play finishers (top-quartile rim rate, no 3s,
    # no self-creation -- Gafford/Capela/Duren types) are systematically OVERRATED
    # by the lineup model: forward residual vs next-season pure RAPM -0.31
    # (t=-2.2, p=.03), offense-specific, and DARKO shows no such bias. Correction
    # fit by walk-forward joint regression arb_{t+1} ~ impact_t + lobScore_t:
    # gamma = -0.23 pts/100 per lobScore unit zeroes the archetype residual OOS
    # (-0.11, p=.35) without hurting overall forward Spearman (.3067 -> .3057).
    # Applied to every offense-bearing impact display (model per-100, hybrid
    # scores, WAA); booker_proj.csv carries its own walk-forward version.
    # (Related null: substitute/backup quality does NOT bias impacts -- resid vs
    # sub-quality corr -0.03, p=.21 -- so no on/off-style correction is needed.)
    LOB_GAMMA = -0.23
    lob_corr = {}
    kpm = 0.00012      # wins-per-(pt/100)-per-minute fallback, refined below
    sqp = CACHE / "shot_quality.csv"
    bfp = HERE / "booker_bookerformer_ratings.csv"
    min_map = {}
    if sqp.exists() and bfp.exists():
        bf = pd.read_csv(bfp)
        min_map = {(int(p), int(s)): float(m)
                   for p, s, m in zip(bf.PLAYER_ID, bf.season, bf.minutes)}
        stable = bf[(bf.impact_total.abs() > 1.0) & (bf.minutes > 500)]
        kpm = float((stable.waa_total / (stable.impact_total * stable.minutes)).median())
        sq = pd.read_csv(sqp)
        sq["minutes"] = [min_map.get((int(p), int(s)), 0.0)
                         for p, s in zip(sq.PLAYER_ID, sq.season)]
        for s, g in sq.groupby("season"):
            pool = g[g.minutes >= 750]
            if len(pool) < 100:
                continue
            def _z(c):
                mu, sd = pool[c].mean(), pool[c].std()
                return (((g[c] - mu) / sd) if sd else g[c] * 0.0).clip(-2.5, 2.5)
            score = (_z("rim_rate") - _z("three_rate") - _z("self_create")).clip(lower=0)
            for pid, v in zip(g.PLAYER_ID, score):
                if pd.notna(v) and v > 0.05:
                    lob_corr[(int(pid), int(s))] = LOB_GAMMA * float(v)

    PER100_FIELDS = ("bookerScore", "bookerOff", "waaOff100", "waaModel100",
                     "waaBayesianOff100", "waaBayesian100",
                     "hybBooker", "hybBkOff", "hybOff", "hybWaaOff100")
    WINS_FIELDS = ("waaOff", "waaModel", "hybWaa", "hybWaaOff")

    def _apply_lob(key, obj):
        d = lob_corr.get(key)
        if not d or not obj:
            return
        dw = d * kpm * min_map.get(key, 1200.0)   # wins-unit equivalent
        for f in PER100_FIELDS:
            if obj.get(f) is not None:
                obj[f] = round(obj[f] + d, 2)
        for f in WINS_FIELDS:
            if obj.get(f) is not None:
                obj[f] = round(obj[f] + dw, 2)

    for key in set(model_map) | set(grades_map):
        _apply_lob(key, model_map.get(key))
        _apply_lob(key, grades_map.get(key))
    if lob_corr:
        n_hit = sum(1 for k in lob_corr if k in model_map or k in grades_map)
        print(f"play-finisher calibration: {n_hit} player-seasons adjusted "
              f"(gamma {LOB_GAMMA}/lob unit, wins conv {kpm:.6f}/min)")
    # per-game on-court net rating (raw observable behind the ratings; Trajectory
    # scatter). Compact int arrays in game order, clipped +-60 at build time.
    gamenet_map = {}
    gnpath = CACHE / "player_game_net.csv"
    if gnpath.exists():
        for r in pd.read_csv(gnpath).itertuples():
            gamenet_map[(int(r.PLAYER_ID), int(r.season))] = \
                [int(x) for x in str(r.game_nets).split("|")]
    # player DNA: continuous style fingerprint (PCA-space) + profile rarity + comparables.
    # dna_seasons[pid] is a sorted list of (season, payload) so we can carry the most
    # recent DNA forward to a season the caches don't cover yet (e.g. 2026).
    dna_seasons = {}
    dpath = CACHE / "player_dna.csv"
    if dpath.exists():
        for r in pd.read_csv(dpath).itertuples():
            comps = []
            for tok in str(r.comparables).split("|"):
                parts = tok.split(":")
                if len(parts) >= 4:
                    comps.append({"pid": int(parts[0]), "player": ":".join(parts[1:-2]),
                                  "season": int(parts[-2]), "sim": float(parts[-1])})
            axes = []
            for tok in str(r.style_fp).split("|"):
                if ":" in tok:
                    a, p = tok.rsplit(":", 1)
                    axes.append({"axis": a, "pct": int(p)})
            dna_seasons.setdefault(int(r.PLAYER_ID), []).append((int(r.season), {
                "styleAxes": axes, "scarcityPct": int(r.scarcity_pct), "comparables": comps}))
        for pid in dna_seasons:
            dna_seasons[pid].sort()

    def _dna_for(pid, season):
        lst = dna_seasons.get(pid)
        if not lst:
            return None
        best = None
        for s, d in lst:
            if s <= season:
                best = d
        return best if best is not None else lst[0][1]
    pos_by_nm, age_map, latest_age, yos = _contract_lookups()
    # The box table lags play-by-play by a season; merge BookerData's backfilled
    # ages so the newest class (and everyone's latest season) shows a real age
    # instead of the 27.0 default.
    for (nm, s), a in data.AGE.items():
        age_map.setdefault((nm, int(s)), float(a))
    latest_seen = {}
    for (nm, s), a in age_map.items():
        if nm not in latest_seen or s > latest_seen[nm][0]:
            latest_seen[nm] = (s, a)
    for nm, (_, a) in latest_seen.items():
        latest_age.setdefault(nm, a)
    proj_min = mm.project_minutes(data, 2027)
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
        # card minutes from the corrected stint feed (BookerFormer ratings file); the
        # legacy base table carries pre-audit stint minutes (~10% low, 2026 partial)
        if key in min_map:
            row["min"] = round(min_map[key])
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
        if int(r.pid) in heights:
            row["heightIn"] = heights[int(r.pid)]
        lc = lctx.get(key)
        if lc:
            row.update({k: lc[k] for k in ("opp_quality", "tm_quality", "real_pm")})
        rv = role_map.get(key)
        if rv:
            row["role"] = rv
        cl = clutch_map.get(int(r.pid))
        if cl:
            row["clutch"] = cl
        gn = gamenet_map.get(key)
        if gn:
            row["gameNets"] = gn
        pjv = proj_map.get(key)
        if pjv is not None:
            row["projNext"] = round(pjv, 2)
        if int(r.pid) in projmin_map:
            row["projMin"] = round(projmin_map[int(r.pid)])
        dn = _dna_for(int(r.pid), int(r.season))
        if dn:
            row["styleAxes"] = dn["styleAxes"]
            row["scarcityPct"] = dn["scarcityPct"]
            if dn["comparables"]:
                row["comparables"] = dn["comparables"]
        gr = grades_map.get(key)
        if gr:
            row["grade"] = gr["grade"]; row["gradeLetter"] = gr["gradeLetter"]
            row["gradeOff"] = gr["gradeOff"]; row["gradeDef"] = gr["gradeDef"]
            if gr.get("proj1") is not None:
                row["gradeProj"] = [{"age": gr[f"proj{t}Age"], "grade": gr[f"proj{t}"],
                                     "letter": gr[f"proj{t}Letter"]} for t in (1, 2, 3)]
            # make offense/defense/WAA/BOOKER HYBRID (50% skills + 50% impact)
            if gr.get("hybBooker") is not None:
                row["bookerScore"] = gr["hybBooker"]; row["bookerOff"] = gr["hybBkOff"]
                row["bookerDef"] = gr["hybBkDef"]
                row["bfOff100"] = gr["hybOff"]; row["bfDef100"] = gr["hybDef"]
                row["waaOff"] = gr["hybWaaOff"]; row["waaDef"] = gr["hybWaaDef"]
                row["waaModel"] = gr["hybWaa"]; row["waa"] = gr["hybWaa"]
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

    # ---- predictive future-season rows: 2027 / 2028 / 2029 (3 years out).
    # Each player's 3-year GRADE projection (aged along the curve) plus their 2026 hybrid
    # BOOKER/WAA aged forward in value units. Static roster (the 2027 cloned roster).
    AGE_Q = -0.06   # BOOKER-unit aging per (age-27)^2 (peak 27)

    def _age_val(v, a0, a1):
        if v is None or a0 is None:
            return v
        return round(v + AGE_Q * ((a1 - 27) ** 2 - (a0 - 27) ** 2), 2)

    fwd = data.PLAYERS.get(2027)
    if fwd is not None:
        abbr = dict(zip(data.TEAMS[2027].TEAM_ID, data.TEAMS[2027].ABBR)) if 2027 in data.TEAMS else {}
        for PROJ in (2027, 2028, 2029):
            t = PROJ - 2026
            for prow in fwd.itertuples():
                pid = int(prow.PLAYER_ID)
                mins = float(prow.MINUTES)
                g26 = grades_map.get((pid, 2026))
                if mins < 250 or not g26 or g26.get("hybBooker") is None or g26.get(f"proj{t}") is None:
                    continue
                a0 = g26.get("age")
                a1 = (a0 + t) if a0 is not None else None
                row = {
                    "pid": pid, "season": PROJ, "player": prow.NAME,
                    "team": abbr.get(prow.TEAM_ID, "?"), "min": round(mins),
                    "projMin": round(mins),
                    "grade": g26[f"proj{t}"], "gradeLetter": g26[f"proj{t}Letter"],
                    "age": g26[f"proj{t}Age"],
                    "bookerScore": _age_val(g26["hybBooker"], a0, a1),
                    "bookerOff": _age_val(g26["hybBkOff"], a0, a1),
                    "bookerDef": _age_val(g26["hybBkDef"], a0, a1),
                    "bfOff100": _age_val(g26["hybOff"], a0, a1),
                    "bfDef100": _age_val(g26["hybDef"], a0, a1),
                    "waaOff": _age_val(g26["hybWaaOff"], a0, a1),
                    "waaDef": _age_val(g26["hybWaaDef"], a0, a1),
                    "waa": _age_val(g26["hybWaa"], a0, a1),
                    "waaModel": _age_val(g26["hybWaa"], a0, a1),
                    "modelType": "projection", "predictive": True,
                }
                # Year-1 rows are BOOKER-PROJ driven (the validated forward model:
                # DARKO-parity overall, best-in-class on team-changers) rather than
                # the aging-curve nowcast. Scale-align PROJ (model per-100) to the
                # hybrid via the player's own hybrid-vs-model gap -- the exact
                # quantity the trajectory's year-1 point plots -- and re-derive WAA
                # from projected minutes so the board's rank blend follows.
                pjv = proj_map.get((pid, 2026)) if PROJ == 2027 else None
                if pjv is not None:
                    m26 = model_map.get((pid, 2026)) or {}
                    anchor = m26.get("waaBayesian100")
                    delta = ((g26["hybBooker"] - anchor)
                             if anchor is not None else 0.0)
                    pb = round(pjv + delta, 2)
                    dv = pb - row["bookerScore"]
                    row["bookerScore"] = pb
                    for f in ("bookerOff", "bookerDef"):
                        if row.get(f) is not None:
                            row[f] = round(row[f] + dv / 2.0, 2)
                    row["waa"] = row["waaModel"] = round(kpm * mins * pb, 2)
                    row["projSource"] = "booker-proj"
                nm = cv.norm_name(prow.NAME)
                lb.attach_contract_fields(row, pos_by_nm, {nm: (a1 or 27.0)}, yos, row["bookerScore"])
                players.append(row)

    by_season = {}
    for row in players:
        by_season.setdefault(row["season"], []).append(row)
    for season, rows in by_season.items():
        # rank by BOOKER (predictive skill rate); fall back to WAA where BOOKER missing
        rows.sort(key=lambda x: (x.get("bookerScore") if x.get("bookerScore") is not None
                                 else x.get("waaModel", x.get("waa", -99)) - 999), reverse=True)
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
    games, player_games = load_games_and_ratings()
    playoff_split = load_playoff_split()
    pred_bracket = load_pred_bracket()
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
        "games": games,
        "playerGames": player_games,
        "playoffSplit": playoff_split,
        "predBracket": pred_bracket,
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
