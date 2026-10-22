"""
Actual vs EXPECTED per player-season (site "xva" charts): box line, minutes, BOOKER, WAA.

"Expected" for season s is strictly point-in-time -- what our models projected BEFORE s
from data through s-1 (walk-forward):
  box      forecast/fantasy_proj.project (Marcel 5/4/3 + regression K + age multiplier),
           cur=False. K / K_pct are re-tuned per target season on targets TUNE0..s-1 only
           (fantasy_proj tunes once on 2019-23); the age multiplier is fit on 2018..s-1.
           Per-game = projected per-minute rate x Marcel mpg.
  minutes  cache/minutes_proj.csv proj_minutes (ridge trained on seasons < s, features
           <= s-1; roster = opening rotation, i.e. the preseason depth chart). 2027 reads
           players_2027.csv MINUTES (minutes_proj + overrides) -- what the site shows.
           Schedule-scaled for the short seasons: 2021 = 72 games (known preseason),
           2020 = realized 70.6 team games (COVID stoppage; schedule-only, no player info).
  BOOKER   cache/booker_proj.csv (BOOKER-PROJ, walk-forward) scale-aligned to the site's
           displayed hybrid score with the player's hybrid-minus-model gap in his latest
           graded season < s (exactly export_dashboard_data's 2027 projected rows);
           no PROJ -> lost-season carry, else the aged hybrid nowcast.
  WAA      kpm x expected minutes x expected BOOKER (kpm as in the exporter).
Actual box = basketball-reference season table (pi.load_box, complete through 2026) and,
for the live season, ESPN per-game rows (cache/espn_box_2027.csv). Actual BOOKER / WAA =
the exporter's displayed values (hybrid + play-finisher correction); none for 2027.

  python -m forecast.expected      # writes cache/expected_vs_actual.csv + expected_validation.json
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import fantasy_proj as fp
from . import player_impacts as pi

CACHE = pi.CACHE
RAPM = CACHE.parent
OUT = CACHE / "expected_vs_actual.csv"
VAL = CACHE / "expected_validation.json"
SEASONS = range(2019, 2028)
ACT_SEASONS = range(SEASONS.start - 1, 2028)   # + 2018 actuals for the 2019 naive baseline
LIVE = 2027
TUNE0 = 2017           # first K-tuning target (box starts 2015 -> 2 prior seasons)
AGE0 = 2018            # first age-fit target
MIN_VAL = 500          # validation: players with >= 500 actual minutes
LOB_GAMMA = -0.23      # export_dashboard_data play-finisher correction
AGE_RISE, AGE_DECL = -0.047, -0.040   # export_dashboard_data BOOKER aging curve
STATS = ["pts", "reb", "ast", "stl", "blk", "tpm", "tov"]
SUF = {"jr", "sr", "ii", "iii", "iv", "v"}


def _curve(a):
    d = a - 27.0
    return (AGE_RISE if d <= 0 else AGE_DECL) * d * d


# --------------------------------------------------------------------------- names -> PLAYER_ID
def _name_maps():
    """per-season and global norm_name -> PLAYER_ID from the stint rosters + ratings."""
    per, glob = {}, {}
    for s in range(2015, LIVE + 1):
        p = CACHE / f"players_{s}.csv"
        if p.exists():
            d = pd.read_csv(p)
            per[s] = {pi.norm_name(n): int(i) for i, n in zip(d.PLAYER_ID, d.NAME)}
            glob.update(per[s])
    bf = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    for i, n in zip(bf.PLAYER_ID, bf.player):
        glob.setdefault(pi.norm_name(n), int(i))
    e = CACHE / f"espn_box_{LIVE}.csv"
    if e.exists():
        d = pd.read_csv(e)
        for i, n in zip(d.PLAYER_ID, d.NAME):
            glob.setdefault(pi.norm_name(n), int(i))
    return per, glob


def _toks(n):
    return [w for w in n.replace(".", "").split() if w not in SUF]


def _resolve(nm, s, per, glob):
    if nm in per.get(s, {}):
        return per[s][nm]
    if nm in glob:
        return glob[nm]
    # bridge: same last name + first-name prefix ("ron holland" ~ "ronald holland"),
    # unique within the season roster
    tk = _toks(nm)
    if len(tk) < 2:
        return None
    cand = {pid for n2, pid in per.get(s, {}).items()
            if len(_toks(n2)) >= 2 and _toks(n2)[-1] == tk[-1]
            and (_toks(n2)[0].startswith(tk[0]) or tk[0].startswith(_toks(n2)[0]))}
    return cand.pop() if len(cand) == 1 else None


# --------------------------------------------------------------------------- actuals
def _actual_box(b, per, glob):
    rows, miss = [], []
    for r in b[b.season.isin(ACT_SEASONS)].itertuples():
        pid = _resolve(r.nm, int(r.season), per, glob)
        if pid is None:
            miss.append((int(r.season), r.playerName, float(r.minutesPlayed)))
            continue
        g = max(float(r.games), 1.0)
        fga, fta = float(r.total_fieldAttempts), float(r.total_ftAttempts)
        rows.append({"PLAYER_ID": pid, "season": int(r.season), "player": r.playerName,
                     "bbref_team": r.team, "g": float(r.games), "min": float(r.minutesPlayed),
                     "pts": r.total_points / g, "reb": r.total_totalRb / g, "ast": r.total_assists / g,
                     "stl": r.total_steals / g, "blk": r.total_blocks / g, "tpm": r.total_threeFg / g,
                     "tov": r.total_turnovers / g,
                     "ts_pct": (r.total_points / (2 * (fga + 0.44 * fta))) if fga + fta > 0 else np.nan,
                     "fg3_pct": (r.total_threeFg / r.total_threeAttempts) if r.total_threeAttempts > 0 else np.nan})
    A = pd.DataFrame(rows)
    # (rare) two bbref rows resolving to one id in a season: keep the bigger one
    A = A.sort_values("min", ascending=False).drop_duplicates(["PLAYER_ID", "season"])
    tot = b[b.season.isin(ACT_SEASONS)].minutesPlayed.sum()
    mm = sum(m for _, _, m in miss)
    match = {"seasons": f"{ACT_SEASONS.start}-{LIVE - 1}", "rows": int(len(b[b.season.isin(ACT_SEASONS)])), "matched": int(len(A) + 0),
             "unmatched": len(miss), "min_share_matched": round(1 - mm / tot, 5),
             "unmatched_top": sorted(miss, key=lambda x: -x[2])[:12]}
    # live season: ESPN per-game rows (already NBA ids)
    p = CACHE / f"espn_box_{LIVE}.csv"
    if p.exists():
        e = pd.read_csv(p)
        e = e[e["min"] > 0]
        if len(e):
            g = e.groupby("PLAYER_ID").agg(player=("NAME", "last"), bbref_team=("TEAM", "last"),
                                           g=("GAME_ID", "nunique"), min=("min", "sum"),
                                           **{k: (k, "sum") for k in STATS + ["fga", "fta", "tpa"]}).reset_index()
            for k in STATS:
                g[k] = g[k] / g.g
            g["ts_pct"] = (g.pts * g.g) / (2 * (g.fga + 0.44 * g.fta)).where(g.fga + g.fta > 0)
            g["fg3_pct"] = (g.tpm * g.g) / g.tpa.where(g.tpa > 0)
            g["season"] = LIVE
            g["g"] = g.g.astype(float)
            A = pd.concat([A, g.drop(columns=["fga", "fta", "tpa"])], ignore_index=True)
    A["mpg"] = A["min"] / A.g.clip(lower=1)
    return A, match


def _site_actuals():
    """(pid, season) -> displayed (bookerScore, waa) as export_dashboard_data computes them,
    plus the model per-100 (anchor) and kpm."""
    bf = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    stable = bf[(bf.impact_total.abs() > 1.0) & (bf.minutes > 500)]
    kpm = float((stable.waa_total / (stable.impact_total * stable.minutes)).median())
    min_map = {(int(p), int(s)): float(m) for p, s, m in zip(bf.PLAYER_ID, bf.season, bf.minutes)}
    lob = {}
    sq = pd.read_csv(CACHE / "shot_quality.csv")
    sq["minutes"] = [min_map.get((int(p), int(s)), 0.0) for p, s in zip(sq.PLAYER_ID, sq.season)]
    for s, g in sq.groupby("season"):
        pool = g[g.minutes >= 750]
        if len(pool) < 100:
            continue
        z = lambda c: (((g[c] - pool[c].mean()) / pool[c].std()) if pool[c].std() else g[c] * 0.0).clip(-2.5, 2.5)
        score = (z("rim_rate") - z("three_rate") - z("self_create")).clip(lower=0)
        for pid, v in zip(g.PLAYER_ID, score):
            if pd.notna(v) and v > 0.05:
                lob[(int(pid), int(s))] = LOB_GAMMA * float(v)
    site = {}
    for r in bf.itertuples():
        k = (int(r.PLAYER_ID), int(r.season))
        site[k] = {"booker": r.booker_score if pd.notna(r.booker_score) else np.nan, "waa": r.waa_total,
                   "m100": r.impact_total}
    gr = pd.read_csv(CACHE / "player_grades.csv")
    grades = {}
    for r in gr.itertuples():
        k = (int(r.PLAYER_ID), int(r.season))
        grades[k] = r
        if pd.notna(r.hybBooker):
            site.setdefault(k, {"m100": np.nan})
            site[k]["booker"] = r.hybBooker
            site[k]["waa"] = r.hybWaa
    for k, v in site.items():
        d = lob.get(k)
        if d:
            v["booker"] = v.get("booker", np.nan) + d
            v["waa"] = v.get("waa", np.nan) + d * kpm * min_map.get(k, 1200.0)
    return site, grades, kpm, lob


# --------------------------------------------------------------------------- expected box
def _tune(b):
    """errors for every (target, K) so K can be chosen walk-forward per season."""
    eK, eKP = {}, {}
    for t in range(TUNE0, max(SEASONS)):
        A = fp._actual(b, t)
        for kk in fp.K_GRID:
            P = fp.project(b, t, {c: kk for c in fp.COUNT}, {"fg": 150, "ft": 150})
            for k in fp.COUNT:
                eK[(t, kk, k)] = fp._err(P, A, k)[0]
        for kp in fp.KP_GRID:
            P = fp.project(b, t, {c: 600 for c in fp.COUNT}, {"fg": kp, "ft": kp})
            for k in fp.MAKES:
                eKP[(t, kp, k)] = fp._err(P, A, k + "_pct")[0]
    return eK, eKP


def _k_for(s, eK, eKP):
    ts = range(TUNE0, s)
    K = {k: min(fp.K_GRID, key=lambda kk: np.mean([eK[(t, kk, k)] for t in ts])) for k in fp.COUNT}
    KP = {k: min(fp.KP_GRID, key=lambda kp: np.mean([eKP[(t, kp, k)] for t in ts])) for k in fp.MAKES}
    return K, KP


def _sched_factor(s):
    if s >= LIVE:
        return 1.0
    g = pd.read_csv(CACHE / f"games_{s}.csv")
    n = int((g.SEASON_TYPE == "Regular Season").sum())
    f = n * 2 / 30 / 82
    return f if f < 0.95 else 1.0      # only the real short seasons (2020, 2021); ignore 1-5 missing games


def _expected_box(b, per, glob):
    eK, eKP = _tune(b)
    out, kinfo = [], {}
    for s in SEASONS:
        K, KP = _k_for(s, eK, eKP)
        age_c = fp.fit_age(b, range(AGE0, s), K, KP)
        bs = b[b.season < s]                         # nothing from s or later
        P = fp.project(bs, s, K, KP, age_c, cur=False)
        kinfo[s] = {"K": K, "KP": KP}
        P["PLAYER_ID"] = [_resolve(n, s, per, glob) or _resolve(n, s - 1, per, glob) for n in P.nm]
        P = P[P.PLAYER_ID.notna()].copy()
        P["PLAYER_ID"] = P.PLAYER_ID.astype(int)
        P["e_mpg"] = P.mpg.clip(8, 38)
        E = pd.DataFrame({"PLAYER_ID": P.PLAYER_ID, "season": s, "e_mpg": P.e_mpg, "e_age": P.age,
                          "e_wmin": P.wmin})
        for k in STATS:
            E["e_" + k] = P[k] * P.e_mpg
        E["e_ts_pct"] = P.pts / (2 * (P.fga + 0.44 * P.fta))
        out.append(E.sort_values("e_wmin", ascending=False).drop_duplicates("PLAYER_ID"))
    return pd.concat(out, ignore_index=True), kinfo


def _expected_minutes():
    m = pd.read_csv(CACHE / "minutes_proj.csv")
    m = m[m.season.isin(SEASONS) & (m.season < LIVE)].copy()
    m["fac"] = m.season.map({s: _sched_factor(s) for s in SEASONS if s < LIVE})
    m["e_min"] = m.proj_minutes * m.fac
    m = m.sort_values("e_min", ascending=False).drop_duplicates(["PLAYER_ID", "season"])
    m = m[["PLAYER_ID", "season", "team", "e_min", "name"]].rename(columns={"team": "e_team"})
    live = pd.read_csv(CACHE / f"players_{LIVE}.csv")
    tm = pd.read_csv(CACHE / f"teams_{LIVE}.csv") if (CACHE / f"teams_{LIVE}.csv").exists() else None
    ab = {}
    if tm is not None and "ABBR" in tm:
        ab = dict(zip(tm.TEAM_ID, tm.ABBR))
    mp = pd.read_csv(CACHE / "minutes_proj.csv")
    mp = mp[mp.season == LIVE].drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").team
    L = pd.DataFrame({"PLAYER_ID": live.PLAYER_ID.astype(int), "season": LIVE, "e_min": live.MINUTES.astype(float),
                      "name": live.NAME})
    L["e_team"] = [mp.get(p, ab.get(t)) for p, t in zip(live.PLAYER_ID, live.TEAM_ID)]
    return pd.concat([m, L], ignore_index=True)


# --------------------------------------------------------------------------- expected BOOKER
def _expected_booker(site, grades, lob):
    proj = pd.read_csv(CACHE / "booker_proj.csv")
    pj = {(int(p), int(s)): float(v) for p, s, v in zip(proj.PLAYER_ID, proj.proj_for_season, proj.proj_impact)}
    carry = pd.read_csv(CACHE / "rating_carry.csv")
    cy = {(int(p), int(s)): float(v) for p, s, v in zip(carry.PLAYER_ID, carry.season, carry.impact_total)}
    e = pd.read_csv(RAPM / "booker_waa_enhanced_ratings.csv")
    enh = set(zip(e.pid.astype(int), e.season.astype(int)))
    graded = {}
    for (pid, s), g in grades.items():
        if pd.notna(g.hybBooker):
            graded.setdefault(pid, []).append(s)
    pids = {p for p, _ in pj} | set(graded)
    rows = []
    for s in SEASONS:
        for pid in pids:
            gss = [x for x in graded.get(pid, []) if x < s]
            gs = max(gss) if gss else None
            g = grades.get((pid, gs)) if gs is not None else None
            a0 = g.age if g is not None and pd.notna(g.age) else None
            ti = (s - gs) if gs is not None else None
            a1 = (a0 + ti) if a0 is not None else None
            val, src = pj.get((pid, s)), "proj"
            if val is None and (pid, s - 1) in cy and g is not None:
                val = cy[(pid, s - 1)] + (_curve(a1) - _curve(a1 - 1) if a1 is not None else 0.0)
                src = "carry"
            if val is not None:
                anchor = site.get((pid, gs), {}).get("m100", np.nan) if gs is not None else np.nan
                if (pid, gs) not in enh:
                    # exporter quirk, mirrored so 2027 equals the site: leaderboard_data's
                    # model_map row for a player with no enhanced-ridge row carries no
                    # waaBayesian100 -> anchor None -> NO hybrid gap (raw PROJ shown)
                    anchor = np.nan
                if g is not None and pd.notna(anchor):
                    val = val + (g.hybBooker - anchor)    # lob correction cancels in the gap
                rows.append((pid, s, val, src))
                continue
            # aged hybrid nowcast (the exporter's non-PROJ fallback; it also requires a grade
            # projection at that horizon -- mirrored for the live season only, since the
            # historical grade tables carry proj1..4 for few rows; 2019 has no PROJ at all)
            if g is None or ti > 4 or (s == LIVE and pd.isna(getattr(g, f"proj{ti}", np.nan))):
                continue
            val = g.hybBooker + lob.get((pid, gs), 0.0) + ((_curve(a1) - _curve(a0)) if a0 is not None else 0.0)
            rows.append((pid, s, val, "aged"))
    return pd.DataFrame(rows, columns=["PLAYER_ID", "season", "e_booker", "e_booker_src"])


# --------------------------------------------------------------------------- validation
def _metrics(y, p):
    ok = y.notna() & p.notna()
    y, p = y[ok].astype(float), p[ok].astype(float)
    if len(y) < 20:
        return None
    return {"n": int(len(y)), "mae": round(float(np.mean(np.abs(p - y))), 4),
            "rmse": round(float(np.sqrt(np.mean((p - y) ** 2))), 4),
            "corr": round(float(np.corrcoef(p, y)[0, 1]), 4), "bias": round(float(np.mean(p - y)), 4)}


def validate(D, prev):
    stats = ["mpg", "min", "g"] + STATS + ["ts_pct", "booker", "waa"]
    last = prev.set_index(["PLAYER_ID", "season"])
    res = {}
    for s in sorted(D.season.unique()):
        if s >= LIVE:
            continue
        d = D[(D.season == s) & (D["min"] >= MIN_VAL)].copy()
        for k in stats:
            nv = [last[k].get((p, s - 1), np.nan) for p in d.PLAYER_ID]
            d["naive_" + k] = nv
            both = d["e_" + k].notna() & d["naive_" + k].notna()
            m, n0 = _metrics(d[k][both], d["e_" + k][both]), _metrics(d[k][both], d["naive_" + k][both])
            if m:
                res.setdefault(k, {})[int(s)] = {"model": m, "naive": n0,
                                                 "coverage": round(float(d["e_" + k].notna().mean()), 3)}
    # leakage check: a preseason forecast built from s-1 must track s-1 at least as well as
    # it tracks s (its inputs); corr(e, actual_s) >> corr(e, actual_s-1) would flag a leak
    leak = {}
    for k in stats:
        d = D[(D.season < LIVE) & (D["min"] >= MIN_VAL)]
        prev = pd.Series([last[k].get((p, s - 1), np.nan) for p, s in zip(d.PLAYER_ID, d.season)], index=d.index)
        ok = d["e_" + k].notna() & prev.notna() & d[k].notna()
        if ok.sum() > 50:
            leak[k] = {"corr_e_actual": round(float(np.corrcoef(d["e_" + k][ok], d[k][ok])[0, 1]), 4),
                       "corr_e_prev": round(float(np.corrcoef(d["e_" + k][ok], prev[ok])[0, 1]), 4),
                       "corr_prev_actual": round(float(np.corrcoef(prev[ok], d[k][ok])[0, 1]), 4)}
    pooled = {}
    for k, v in res.items():
        ss = list(v)
        w = np.array([v[s]["model"]["n"] for s in ss])
        pooled[k] = {f: round(float(np.average([v[s][src][f] for s in ss], weights=w)), 4)
                     for src in ("model",) for f in ("mae", "rmse", "corr")}
        pooled[k].update({"naive_" + f: round(float(np.average([v[s]["naive"][f] for s in ss], weights=w)), 4)
                          for f in ("mae", "rmse", "corr")})
        pooled[k]["mae_gain_pct"] = round(100 * (1 - pooled[k]["mae"] / pooled[k]["naive_mae"]), 1)
        pooled[k]["n"] = int(w.sum())
    return {"by_season": res, "pooled": pooled, "leak_check": leak}


# --------------------------------------------------------------------------- build
def build(write=True):
    b = fp._box()
    print("bbref box seasons:", b.groupby("season").minutesPlayed.sum().round(-3).astype(int).to_dict())
    per, glob = _name_maps()
    A, match = _actual_box(b, per, glob)
    print(f"name match {ACT_SEASONS.start}-{LIVE - 1}: {match['matched']}/{match['rows']} rows, "
          f"{100 * match['min_share_matched']:.2f}% of minutes; unmatched top: {match['unmatched_top'][:6]}")
    site, grades, kpm, lob = _site_actuals()
    A["booker"] = [site.get((p, s), {}).get("booker", np.nan) if s < LIVE else np.nan
                   for p, s in zip(A.PLAYER_ID, A.season)]
    A["waa"] = [site.get((p, s), {}).get("waa", np.nan) if s < LIVE else np.nan
                for p, s in zip(A.PLAYER_ID, A.season)]
    EB, kinfo = _expected_box(b, per, glob)
    EM = _expected_minutes()
    EK = _expected_booker(site, grades, lob)
    E = EM.merge(EB, on=["PLAYER_ID", "season"], how="outer").merge(EK, on=["PLAYER_ID", "season"], how="outer")
    # keep expected-only rows only if they had a real preseason role (opening roster / PROJ)
    E = E[E.e_min.notna() | (E.e_booker_src == "proj")]
    D = A.merge(E, on=["PLAYER_ID", "season"], how="outer")
    D = D[D.season.isin(SEASONS)]
    D["e_g"] = np.minimum(D.e_min / D.e_mpg, 82 * D.season.map(_sched_factor))
    D["e_waa"] = kpm * D.e_min * D.e_booker
    # identity
    bfr = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    tm = {(int(p), int(s)): t for p, s, t in zip(bfr.PLAYER_ID, bfr.season, bfr.team)}
    nm = {}
    for s in range(2015, LIVE + 1):
        p = CACHE / f"players_{s}.csv"
        if p.exists():
            d = pd.read_csv(p)
            nm.update({int(i): n for i, n in zip(d.PLAYER_ID, d.NAME)})
    D["player"] = [pl if isinstance(pl, str) else (n2 if isinstance(n2, str) else nm.get(int(p), str(p)))
                   for pl, n2, p in zip(D.player, D.name, D.PLAYER_ID)]
    D["team"] = [tm.get((int(p), int(s))) or (bt if isinstance(bt, str) and not re.fullmatch(r"\d+TM", bt) else et)
                 for p, s, bt, et in zip(D.PLAYER_ID, D.season, D.bbref_team, D.e_team)]
    D.loc[D.season == LIVE, "team"] = [bt if isinstance(bt, str) else et for bt, et in
                                       zip(D.loc[D.season == LIVE, "bbref_team"], D.loc[D.season == LIVE, "e_team"])]
    for c in ("g", "min"):
        D[c] = D[c].fillna(0.0)          # expected but never played (season lost)
    D.loc[D.g == 0, "mpg"] = 0.0
    cols = (["PLAYER_ID", "season", "player", "team", "g", "mpg", "min"] + STATS + ["ts_pct", "fg3_pct", "booker", "waa",
            "e_g", "e_mpg", "e_min"] + ["e_" + k for k in STATS] + ["e_ts_pct", "e_booker", "e_waa", "e_booker_src"])
    D = D[cols].sort_values(["season", "min"], ascending=[True, False]).reset_index(drop=True)
    D["PLAYER_ID"] = D.PLAYER_ID.astype(int)
    D["season"] = D.season.astype(int)
    V = validate(D, pd.concat([A[A.season < SEASONS.start], D], ignore_index=True))
    V.update({"match": match, "kpm": kpm, "k_by_season": {int(s): v for s, v in kinfo.items()},
              "min_actual_minutes": MIN_VAL,
              "sched_factor": {int(s): round(_sched_factor(s), 4) for s in SEASONS},
              "notes": "expected = walk-forward preseason projection from data through s-1; naive = player's "
                       "actual value in s-1; per-game stats; booker/waa on the site's displayed (hybrid) scale"})
    print("\npooled (2019-26, >= 500 actual min) model vs naive last-season:")
    print(pd.DataFrame(V["pooled"]).T[["n", "mae", "naive_mae", "mae_gain_pct", "corr", "naive_corr"]].to_string())
    print("\nleak check (corr of expected with actual s / with actual s-1 / naive s-1 vs s):")
    print(pd.DataFrame(V["leak_check"]).T.to_string())
    if write:
        D.round(4).to_csv(OUT, index=False)
        VAL.write_text(json.dumps(V, indent=1, default=float))
        print(f"wrote {OUT} ({len(D)} rows) and {VAL}")
    return D, V


if __name__ == "__main__":
    build()
