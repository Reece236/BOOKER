"""
TRUE-SKILL rate stats from play-by-play.

The master box dataset only carries raw per-32 counting stats (and stops at 2025).
Here we mine the nbastats PBP directly for every season (incl. 2026) and turn the raw
counts into empirical-Bayes-shrunk TRUE-SKILL rates per 36 minutes -- the leaderboard
should rank a player's talent-level rate, not a small-sample counting-stat fluke.

Attribution (verified against descriptions):
  assists  EVENTMSGTYPE==1 (made FG)      -> PLAYER2 (the passer); "3PT" in text = a 3 created
  steals   EVENTMSGTYPE==5 (turnover)     -> PLAYER2 when description says "STEAL"
  blocks   EVENTMSGTYPE==2 (missed FG)    -> PLAYER3 (the blocker)
  rebounds EVENTMSGTYPE==4, PLAYER1>0     -> PLAYER1 (team rebounds have PLAYER1_ID==0)

Rates are per 36 min (minutes from BookerData.PLAYERS). Each rate is shrunk toward the
league mean by its Poisson sampling variance (Var(rate) ~ rate^2 / count): a full-season
regular barely moves, a 200-minute sample collapses toward average.

Output cache/pbp_skills.csv per (season, PLAYER_ID):
  minutes, ast/stl/blk/reb counts, raw per-36 rates (ast36 ...),
  true-skill per-36 rates (true_ast36 ...), and create36 (expected pts created via
  assists, per 36) + its shrunk true_create36.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
OUT = CACHE / "pbp_skills.csv"
MIN_QUAL = 500          # minutes to qualify for estimating each rate's population spread


def _eb_shrink(rate, svar, qual):
    """Normal-normal empirical-Bayes posterior mean; shrink each rate toward the
    qualified-player mean by its sampling variance (tau^2 = Var(rate) - mean svar)."""
    rate = np.asarray(rate, float)
    svar = np.asarray(svar, float)
    q = np.asarray(qual, bool)
    if q.sum() < 8:
        q = np.isfinite(svar)
    mu = float(np.mean(rate[q]))
    tau2 = max(0.0, float(np.var(rate[q]) - np.mean(svar[q])))
    shrink = tau2 / (tau2 + np.where(np.isfinite(svar), svar, np.inf))
    return mu + shrink * (rate - mu)


def _alltext(d):
    return (d.HOMEDESCRIPTION.fillna("") + " " + d.VISITORDESCRIPTION.fillna("")
            + " " + d.NEUTRALDESCRIPTION.fillna(""))


_AST_RE = None


def _counts_from_parquet(season):
    """Fallback when the legacy nbastats feed is unavailable (e.g. 2025-26): reconstruct
    per-player counts from the modern CDN play-by-play parquet (cache/pbp_{season}/).
    Recoverable: assists/created-points, rebounds, FGA, FT (make/miss), turnovers, fouls
    -- the assister is parsed from the shot description by last name (mapped per game).
    NOT recoverable from this offense-oriented feed: steals & blocks (no defender
    attribution) -- returned as NaN and carried forward from the prior season in build()."""
    import glob
    import re
    from collections import defaultdict
    files = sorted(glob.glob(str(CACHE / f"pbp_{season}" / "*.parquet")))
    if not files:
        return None
    global _AST_RE
    if _AST_RE is None:
        _AST_RE = re.compile(r"\(([A-Za-z.''\-]+(?: [A-Za-z.''\-]+)?) \d+ AST\)")
    acc = defaultdict(lambda: defaultdict(float))
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        roster = {}                                        # last name -> personId (this game)
        for pid, nm in zip(d.personId, d.playerName):
            if pid and pid > 0 and isinstance(nm, str):
                roster.setdefault(nm, int(pid))
        for at, pid_, desc, sv in zip(d.actionType, d.personId, d.description.fillna(""), d.shotValue):
            pid = int(pid_) if (pid_ and pid_ > 0) else 0
            if at in ("Made Shot", "Missed Shot") and pid:
                acc[pid]["fga"] += 1
            if at == "Made Shot":
                m = _AST_RE.search(desc)
                if m:
                    nm = m.group(1)
                    ap = roster.get(nm) or roster.get(nm.split()[0]) or roster.get(nm.split()[-1])
                    if ap:
                        is3 = (sv == 3)
                        acc[ap]["ast"] += 1
                        acc[ap]["create"] += 3.0 if is3 else 2.0
                        if is3:
                            acc[ap]["ast3"] += 1
            elif at == "Rebound" and pid:
                acc[pid]["reb"] += 1
            elif at == "Free Throw" and pid:
                acc[pid]["fta"] += 1
                if "Miss" not in desc:
                    acc[pid]["ftm"] += 1
            elif at == "Turnover" and pid:
                acc[pid]["tov"] += 1
            elif at == "Foul" and pid:
                acc[pid]["pf"] += 1
    if not acc:
        return None
    rows = [{"PLAYER_ID": pid, "season": int(season),
             "ast": v["ast"], "ast3": v["ast3"], "create": v["create"],
             "stl": 0.0, "blk": 0.0, "reb": v["reb"], "fga": v["fga"],
             "fta": v["fta"], "ftm": v["ftm"], "tov": v["tov"], "pf": v["pf"]}
            for pid, v in acc.items()]
    print(f"  pbp_skills {season}: reconstructed from CDN parquet "
          f"(steals/blocks carried from prior season)")
    return pd.DataFrame(rows)


def season_counts(season):
    """Per-player raw counts for one season from the PBP. Also returns assisted-shot
    expected points (3s worth 3, 2s worth 2 -- a simple created-shot value)."""
    p = CACHE / f"nbastats_{season}.csv"
    if not p.exists():
        # ESPN full-season fallback (stats.nba.com is IP-blocked here; ESPN is reachable).
        # cache/espn_pbp_{season}.csv carries full-season counts + MINUTES + steals/blocks
        # (the CDN parquet lacked defender attribution). Built by data_ingest/fetch_espn_2026.py.
        espn = CACHE / f"espn_pbp_{season}.csv"
        if espn.exists():
            print(f"  pbp_skills {season}: from ESPN full-season totals")
            return pd.read_csv(espn)
        return None
    d = pd.read_csv(p, low_memory=False)
    txt = _alltext(d)

    asts = d[(d.EVENTMSGTYPE == 1) & (d.PLAYER2_ID > 0)].copy()
    asts["is3"] = txt[asts.index].str.contains("3PT", regex=False)
    ast = asts.groupby("PLAYER2_ID").size()
    ast3 = asts[asts.is3].groupby("PLAYER2_ID").size()
    # expected points created via assists (3->3, 2->2): rewards creating 3-point looks
    asts["cpts"] = np.where(asts.is3, 3.0, 2.0)
    create = asts.groupby("PLAYER2_ID").cpts.sum()

    stl_rows = d[(d.EVENTMSGTYPE == 5) & (d.PLAYER2_ID > 0) & txt.str.contains("STEAL", regex=False)]
    stl = stl_rows.groupby("PLAYER2_ID").size()
    blk = d[(d.EVENTMSGTYPE == 2) & (d.PLAYER3_ID > 0)].groupby("PLAYER3_ID").size()
    reb = d[(d.EVENTMSGTYPE == 4) & (d.PLAYER1_ID > 0)].groupby("PLAYER1_ID").size()

    # FGA (made+missed FG by the shooter), free throws, turnovers committed, fouls committed
    fga = d[(d.EVENTMSGTYPE.isin([1, 2])) & (d.PLAYER1_ID > 0)].groupby("PLAYER1_ID").size()
    ft = d[(d.EVENTMSGTYPE == 3) & (d.PLAYER1_ID > 0)].copy()
    ft["miss"] = txt[ft.index].str.contains("MISS", regex=False)
    fta = ft.groupby("PLAYER1_ID").size()
    ftm = ft[~ft.miss].groupby("PLAYER1_ID").size()
    tov = d[(d.EVENTMSGTYPE == 5) & (d.PLAYER1_ID > 0)].groupby("PLAYER1_ID").size()
    pf = d[(d.EVENTMSGTYPE == 6) & (d.PLAYER1_ID > 0)].groupby("PLAYER1_ID").size()

    out = pd.DataFrame({"ast": ast, "ast3": ast3, "create": create,
                        "stl": stl, "blk": blk, "reb": reb,
                        "fga": fga, "fta": fta, "ftm": ftm, "tov": tov, "pf": pf}).fillna(0.0)
    out.index.name = "PLAYER_ID"
    out["season"] = int(season)
    return out.reset_index()


def _minutes_map(seasons):
    """{(season, PLAYER_ID): minutes} from BookerData (covers all seasons incl. 2026)."""
    from . import player_impacts as pi
    data = pi.BookerData(seasons=range(min(seasons), max(seasons) + 1))
    mm = {}
    for s in seasons:
        pl = data.PLAYERS.get(s)
        if pl is None:
            continue
        for pid, mn in zip(pl.PLAYER_ID, pl.MINUTES):
            mm[(int(s), int(pid))] = float(mn)
    return mm


def build(seasons=range(2015, 2027), min_minutes=200):
    frames = []
    for s in seasons:
        c = season_counts(s)
        if c is not None:
            frames.append(c)
            print(f"  pbp_skills {s}: {len(c)} players")
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    mm = _minutes_map(sorted(raw.season.unique()))
    # prefer minutes carried on the counts frame (ESPN full-season, e.g. 2026); otherwise
    # BookerData. Keeps ESPN counts and ESPN minutes consistent (BookerData's 2026 is partial).
    carried = raw["minutes"] if "minutes" in raw.columns else pd.Series(np.nan, index=raw.index)
    bd = pd.Series([mm.get((int(s), int(p)), 0.0) for s, p in zip(raw.season, raw.PLAYER_ID)],
                   index=raw.index)
    raw["minutes"] = carried.where(carried.notna() & (carried > 0), bd)
    raw = raw[raw.minutes >= min_minutes].copy()

    u = raw.minutes.values / 36.0                         # 36-min units
    # per-36 Poisson-rate skills (fta36 = foul drawing, pf36 = fouls committed)
    rates = {"ast36": "ast", "stl36": "stl", "blk36": "blk", "reb36": "reb",
             "create36": "create", "fta36": "fta", "pf36": "pf"}
    for rcol, ccol in rates.items():
        raw[rcol] = (raw[ccol].values / u).round(3)
    # proportion skills: turnover rate (TOV / offensive plays) and free-throw %
    plays = raw.fga + 0.44 * raw.fta + raw.tov
    raw["tov_pct"] = (raw.tov / plays.replace(0, np.nan)).round(4)
    raw["ft_pct"] = (raw.ftm / raw.fta.replace(0, np.nan)).round(4)

    def shrink_prop(num, den, qual_n, prior):
        """Binomial EB-shrink of a proportion num/den toward the qualified mean."""
        den = den.astype(float)
        p = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        q = den >= qual_n
        mu = float(np.nanmean(p[q])) if q.any() else prior
        p = np.where(np.isnan(p), mu, p)
        svar = np.where(den > 0, p * (1 - p) / np.maximum(den, 1), np.inf)
        return np.round(_eb_shrink(p, svar, q), 4)

    out_parts = []
    for s, g in raw.groupby("season"):
        g = g.copy()
        uu = g.minutes.values / 36.0
        for rcol, ccol in rates.items():
            cnt = g[ccol].values.astype(float)
            rate = cnt / uu
            svar = (cnt + 1.0) / uu ** 2     # Poisson; +1 keeps zero-count players finite
            g["true_" + rcol] = np.round(_eb_shrink(rate, svar, g.minutes.values >= MIN_QUAL), 3)
        pl = (g.fga + 0.44 * g.fta + g.tov).values
        g["true_tov_pct"] = shrink_prop(g.tov.values, pl, 300, 0.13)     # lower = better
        g["true_ft_pct"] = shrink_prop(g.ftm.values, g.fta.values, 50, 0.77)
        out_parts.append(g)
    out = pd.concat(out_parts, ignore_index=True)
    cols = (["season", "PLAYER_ID", "minutes", "ast", "ast3", "stl", "blk", "reb", "create",
             "fga", "fta", "ftm", "tov", "pf"] + list(rates.keys()) + ["tov_pct", "ft_pct"]
            + ["true_" + r for r in rates] + ["true_tov_pct", "true_ft_pct"])
    out = out[cols]
    # seasons rebuilt from the CDN parquet fallback have no steal/block attribution;
    # carry each player's most-recent prior true steal/block rate forward so those bars
    # aren't zeroed. (Detected as seasons whose raw stl+blk counts are all zero.)
    by_season_def = out.groupby("season")[["stl", "blk"]].sum()
    fallback_seasons = [int(s) for s in by_season_def.index if by_season_def.loc[s].sum() == 0]
    for fs in fallback_seasons:
        prior = out[out.season < fs].sort_values("season")
        if prior.empty:
            continue
        last = prior.groupby("PLAYER_ID")[["stl36", "blk36", "true_stl36", "true_blk36"]].last()
        m = out.season == fs
        for col in ("stl36", "blk36", "true_stl36", "true_blk36"):
            out.loc[m, col] = out.loc[m, "PLAYER_ID"].map(last[col]).where(
                out.loc[m, "PLAYER_ID"].isin(last.index), out.loc[m, col])
        print(f"  pbp_skills {fs}: carried steals/blocks forward for "
              f"{int(m.sum())} players (CDN feed lacks defender attribution)")
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} player-seasons, {out.season.min()}-{out.season.max()})")
    return out


if __name__ == "__main__":
    build()
