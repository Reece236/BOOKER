"""
Sticky, team-independent, age-curved player GRADES (+ 3-year projections).

A grade blends two views of a player and then SMOOTHS across seasons:
  * BOOKER  -- holistic impact value (offense/defense), but on/off-based, so it
    carries RAPM teammate leakage (team-dependent).
  * Skill composite -- the LEAKAGE-FREE individual true-skills (a player's own
    shooting / playmaking / rim-protection events; difficulty-adjusted).

  grade_z = 0.5 * z(BOOKER) + 0.5 * z(skill composite)

Multi-year empirical smoothing (trailing decay) makes the grade STICKY and, because
the leaked half is averaged across changing teammates/roles, substantially
TEAM-INDEPENDENT. The grade is then read on a 0-100 scale and projected 1/2/3 seasons
ahead along the NBA aging curve (peak 27, steeper decline after).

Output cache/player_grades.csv per (season, PLAYER_ID):
  age, grade, gradeLetter, gradeOff, gradeDef   (sticky, current)
  and on each player's LATEST season: proj{1,2,3} grade + letter + age.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
OUT = CACHE / "player_grades.csv"

AGE_PEAK = 27.0
SMOOTH = [1.0, 0.6, 0.36]   # trailing-season weights (stickiness + de-leaking)
# Per-possession SKILL dims as (column, WEIGHT) -- ALL are per-36/per-100 RATES, so the
# grade is independent of MINUTES. Weight sign = direction ("lower is better" is negative).
#
# Offensive weights are LEARNED from predicting a player's NEXT-season impact_off out of
# sample (ridge, walk-forward) -- and the ordering is exactly what the Basketball Bible's
# "it starts on the shot" implies: shot QUALITY + SELECTION dominate. Shot-making, getting
# to the RIM, drawing FTs, and THREE-rate carry the most predictive weight; playmaking and
# spacing add moderate weight; turnovers subtract. Raw make-% (true_3p/true_rim/true_efg,
# collinear with shot_making) and raw shot VOLUME (fga36) earned ~zero OOS weight, so they
# are dropped -- volume is not value. This weighting beat the old equal-weight composite
# out of sample (next-impact R2 .801 -> .833). Defense stayed ~equal (learned ~ equal,
# R2 .772 -> .777), so DEF_SKILLS is left equal-weighted rather than over-fit.
OFF_SKILLS = [("shot_making", 0.43), ("rim_rate", 0.35), ("true_fta36", 0.32),
              ("three_rate", 0.18), ("true_create36", 0.17), ("space_threat", 0.15),
              ("true_ast36", 0.15), ("true_tov_pct", -0.13), ("self_create", 0.10),
              ("off_gravity", 0.06), ("on_gravity", 0.02)]
DEF_SKILLS = [("true_blk36", 1), ("true_stl36", 1), ("true_reb36", 1), ("true_pf36", -1)]
K_GRADE = 150   # minutes at which a season grade is shrunk 50% toward the league mean
GRADE_LETTERS = [(97, "A+"), (90, "A"), (82, "A-"), (73, "B+"), (62, "B"),
                 (50, "B-"), (38, "C+"), (27, "C"), (18, "C-"), (9, "D"), (0, "F")]


def _letter(g100):
    for thresh, lab in GRADE_LETTERS:
        if g100 >= thresh:
            return lab
    return "F"


def age_curve_z(age):
    """Aging curve in grade-z units (0 at peak 27; gentler rise, steeper decline)."""
    d = age - AGE_PEAK
    return -0.018 * d * d if d <= 0 else -0.045 * d * d


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def build(seasons=range(2018, 2027)):
    from . import contract_value as cv
    rat = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    rat = rat[rat.season.isin(seasons)].copy()
    sq = pd.read_csv(CACHE / "shot_quality.csv")
    pp = pd.read_csv(CACHE / "pbp_skills.csv")
    sk = sq.merge(pp, on=["season", "PLAYER_ID"], how="outer")
    gpath = CACHE / "gravity.csv"
    if gpath.exists():
        sk = sk.merge(pd.read_csv(gpath), on=["season", "PLAYER_ID"], how="left")
    # offensive role term: FGA per 36 (minutes-INDEPENDENT). shots from shot_quality;
    # minutes from ratings (cover all grade seasons incl. 2026), falling back to pbp.
    sk = sk.merge(rat[["season", "PLAYER_ID", "minutes"]].rename(columns={"minutes": "rmin"}),
                  on=["season", "PLAYER_ID"], how="left")
    mins = sk["rmin"].fillna(sk["minutes"]) if "minutes" in sk else sk["rmin"]
    sk["fga36"] = sk["shots"] / (mins.replace(0, np.nan) / 36.0)
    age_map, _ = cv._player_ages()
    # age anchors (most recent known age per player) -> impute ages for seasons the
    # master box doesn't cover yet (e.g. 2026): age = anchor_age + season offset.
    anchor = {}
    for (nm, ss), a in age_map.items():
        if nm not in anchor or ss > anchor[nm][0]:
            anchor[nm] = (ss, a)

    def age_of(nm, s):
        v = age_map.get((nm, int(s)))
        if v is not None:
            return float(v)
        if nm in anchor:
            return float(anchor[nm][1] + (s - anchor[nm][0]))
        return None

    # per-season z-scores of skills + BOOKER, then the blended grade_z
    parts = []
    for s, g in sk.groupby("season"):
        g = g.copy()
        # unsigned per-skill z (weight carries sign); missing skill -> neutral (0)
        for c, _w in OFF_SKILLS + DEF_SKILLS:
            g["z_" + c] = (_z(g[c]) if c in g.columns
                           else pd.Series(0.0, index=g.index)).fillna(0.0)

        def _wmean(items):
            num = sum(w * g["z_" + c] for c, w in items)
            den = sum(abs(w) for c, w in items)
            return num / den if den else num * 0.0
        g["off_skill"] = _wmean(OFF_SKILLS)   # weighted (predictive) composite
        g["def_skill"] = _wmean(DEF_SKILLS)   # equal-weighted
        parts.append(g[["season", "PLAYER_ID", "off_skill", "def_skill"]])
    skz = pd.concat(parts, ignore_index=True)

    rows = []
    for s, g in rat.groupby("season"):
        g = g.merge(skz[skz.season == s], on=["season", "PLAYER_ID"], how="left")
        g["bk_off_z"], g["bk_def_z"] = _z(g.booker_off), _z(g.booker_def)
        g["bk_z"] = _z(g.booker_score)
        # skill composite z (fall back to BOOKER where a skill view is missing)
        g["off_skill"] = g["off_skill"].fillna(g["bk_off_z"])
        g["def_skill"] = g["def_skill"].fillna(g["bk_def_z"])
        g["skill_z"] = _z(g.off_skill + g.def_skill)
        g["skill_off_z"], g["skill_def_z"] = _z(g.off_skill), _z(g.def_skill)
        # hybrid blend (50% holistic impact, 50% leakage-free skills)
        g["grade_raw"] = 0.5 * g.bk_z + 0.5 * g.skill_z
        g["grade_off_raw"] = 0.5 * g.bk_off_z + 0.5 * g.skill_off_z
        g["grade_def_raw"] = 0.5 * g.bk_def_z + 0.5 * g.skill_def_z
        # ---- hybrid VALUE metrics (off/def/WAA/BOOKER) in the impact units, per-season:
        # map each skill composite to that metric's scale, then blend 50/50 with impact.
        def mapsk(z, base):
            return base.mean() + z * (base.std() or 1.0)
        g["hybOff"] = 0.5 * g.impact_off + 0.5 * mapsk(g.skill_off_z, g.impact_off)
        g["hybDef"] = 0.5 * g.impact_def + 0.5 * mapsk(g.skill_def_z, g.impact_def)
        g["hybTot"] = g.hybOff + g.hybDef
        g["hybBkOff"] = 0.5 * g.booker_off + 0.5 * mapsk(g.skill_off_z, g.booker_off)
        g["hybBkDef"] = 0.5 * g.booker_def + 0.5 * mapsk(g.skill_def_z, g.booker_def)
        g["hybBooker"] = g.hybBkOff + g.hybBkDef
        presk = (g.waa_off.abs() + g.waa_def.abs() + g.waa_total.abs()) / \
                (g.impact_off.abs() + g.impact_def.abs() + g.impact_total.abs() + 1e-9)
        g["hybWaaOff"] = g.hybOff * presk; g["hybWaaDef"] = g.hybDef * presk
        g["hybWaa"] = g.hybTot * presk
        # Bayesian shrinkage toward the league mean by sample reliability: a low-minute
        # (uncertain) player regresses to average; a full-time player keeps his estimate.
        # This is minutes-INDEPENDENT for well-sampled players -- it only down-weights noise.
        rel = g["minutes"] / (g["minutes"] + K_GRADE)
        for c in ("grade_raw", "grade_off_raw", "grade_def_raw"):
            g[c] = g[c] * rel
        rows.append(g[["season", "PLAYER_ID", "player", "minutes",
                       "grade_raw", "grade_off_raw", "grade_def_raw",
                       "hybOff", "hybDef", "hybTot", "hybBkOff", "hybBkDef", "hybBooker",
                       "hybWaaOff", "hybWaaDef", "hybWaa"]])
    G = pd.concat(rows, ignore_index=True)

    # multi-year trailing smoothing -> sticky, de-leaked
    G = G.sort_values(["PLAYER_ID", "season"])
    gi = {pid: gg.set_index("season") for pid, gg in G.groupby("PLAYER_ID")}
    out = []
    for pid, gg in gi.items():
        for s in gg.index:
            sm = {}
            for col in ("grade_raw", "grade_off_raw", "grade_def_raw"):
                num = den = 0.0
                for t, w in enumerate(SMOOTH):
                    if (s - t) in gg.index:
                        v = gg.loc[s - t, col]
                        if pd.notna(v):
                            num += w * v; den += w
                sm[col] = num / den if den else 0.0
            nm = cv.norm_name(gg.loc[s, "player"])
            age = age_of(nm, int(s))
            out.append({"season": int(s), "PLAYER_ID": int(pid),
                        "player": gg.loc[s, "player"], "minutes": int(gg.loc[s, "minutes"]),
                        "age": age, "gz": sm["grade_raw"],
                        "gz_off": sm["grade_off_raw"], "gz_def": sm["grade_def_raw"]})
    D = pd.DataFrame(out)
    # attach the per-season hybrid VALUE metrics (off/def/WAA/BOOKER) from G
    hcols = ["hybOff", "hybDef", "hybTot", "hybBkOff", "hybBkDef", "hybBooker",
             "hybWaaOff", "hybWaaDef", "hybWaa"]
    hyb = G[["season", "PLAYER_ID"] + hcols].copy()
    for c in hcols:
        hyb[c] = hyb[c].round(2)
    D = D.merge(hyb, on=["season", "PLAYER_ID"], how="left")

    def to100(z):
        return int(np.clip(round(50 + 15 * z), 1, 99))
    D["grade"] = D.gz.map(to100)
    D["gradeOff"] = D.gz_off.map(to100)
    D["gradeDef"] = D.gz_def.map(to100)
    D["gradeLetter"] = D.grade.map(_letter)

    # 3-year projections from each player's LATEST season, aged along the curve
    proj = {}
    for pid, gg in D.groupby("PLAYER_ID"):
        last = gg.loc[gg.season.idxmax()]
        a = last.age
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        base = last.gz - age_curve_z(a)        # peak-anchored talent (remove current-age effect)
        for t in (1, 2, 3):
            za = base + age_curve_z(a + t)
            proj[(int(pid), t)] = (to100(za), _letter(to100(za)), round(float(a + t), 0))
    D["isLatest"] = D.season == D.groupby("PLAYER_ID").season.transform("max")
    for t in (1, 2, 3):
        D[f"proj{t}"] = [proj.get((p, t), (None, None, None))[0] if lt else None
                         for p, lt in zip(D.PLAYER_ID, D.isLatest)]
        D[f"proj{t}Letter"] = [proj.get((p, t), (None, None, None))[1] if lt else None
                               for p, lt in zip(D.PLAYER_ID, D.isLatest)]
        D[f"proj{t}Age"] = [proj.get((p, t), (None, None, None))[2] if lt else None
                            for p, lt in zip(D.PLAYER_ID, D.isLatest)]
    D = D.drop(columns=["gz", "gz_off", "gz_def", "isLatest"])
    D.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(D)} player-seasons, {D.season.min()}-{D.season.max()})")
    return D


if __name__ == "__main__":
    build()
