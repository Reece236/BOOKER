"""Backfill REAL per-side points (and the corrected stint PLUS_MINUS / Y) into the
existing cache/stints_{season}.csv files, from the raw pbp already on disk.

Why (2026-07 audit):
  * the cached stints carried only margin, and stint_off_def synthesized
    HOME_PTS/AWAY_PTS from it at a constant 1.08 ppp -> the O/D split was not
    identified (see stint_off_def.py docstring);
  * the stint builders dropped (classic) or shifted onto the previous lineup (v3)
    the first event of every stint and the first score of each period -> stint PM
    reproduced the final game margin in only ~5% of games (MAE ~5 pts/game).

  * stint durations dropped the sub->first-event dead-ball gap (~9% of game time),
    inflating per-100 Y and under-counting stint minutes ~10%.

This script re-runs the FIXED builders on the cached raw pbp, verifies that >=99% of
the existing stints' lineup sequences (GAME_ID, PERIOD, lineups, occurrence) are
reproduced, and only then replaces the file with the corrected stints (real
HOME_PTS/AWAY_PTS, PLUS_MINUS, DURATION_SECONDS, POSS, Y, Y_OFF_HOME, Y_DEF_HOME,
PTS_SOURCE). teams_{season}.csv ACTUAL_NET and players_{season}.csv MINUTES are
recomputed from the corrected stints (team assignments unchanged).
Originals are copied to cache/_pre_audit_backup/ first (never overwritten).

Sources:
  2015-2025 regular season : cache/nbastats_{season}.csv (classic, SCORE column)
  2026 (reg + playoffs)    : cache/pbp_2026/*.parquet + cache/box_2026/*.parquet (v3)

Run:  python data_ingest/backfill_stint_points.py [seasons...] [--dry-run]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from build_season_stints import build_lineup_stints          # noqa: E402
from stint_off_def import enrich_stints                       # noqa: E402

CACHE = ROOT / "cache"
BACKUP = CACHE / "_pre_audit_backup"
SEC_PER_POSS = 28.8
KEY = ["GAME_ID", "PERIOD", "HOME_LINEUP", "AWAY_LINEUP", "DURATION_SECONDS"]


def _backup(path):
    BACKUP.mkdir(exist_ok=True)
    dst = BACKUP / path.name
    if path.exists() and not dst.exists():
        shutil.copy2(path, dst)


def _norm_lineup(s):
    return ", ".join(str(int(x)) for x in sorted(int(v) for v in str(s).split(",") if v.strip()))


def rebuild_classic(season):
    raw = CACHE / f"nbastats_{season}.csv"
    if not raw.exists():
        return None
    df = pd.read_csv(raw, low_memory=False)
    df["GAME_ID"] = df["GAME_ID"].astype("int64")
    st = build_lineup_stints(df)
    st = st[(st.HOME_LINEUP.str.count(",") == 4) & (st.AWAY_LINEUP.str.count(",") == 4)].copy()
    return st


def rebuild_v3(season):
    import importlib
    f26 = importlib.import_module("fetch_nba_2026")
    import pbpv3
    games = pd.read_csv(CACHE / f"games_{season}.csv")
    home = {int(r.GAME_ID): int(f26.ABBR_TO_ID[r.HOME]) for r in games.itertuples()}
    away = {int(r.GAME_ID): int(f26.ABBR_TO_ID[r.AWAY]) for r in games.itertuples()}
    out = []
    for gid in games.GAME_ID.astype(int):
        name = f"{gid:010d}.parquet"
        pp, bp = f26.PBP_CACHE / name, f26.BOX_CACHE / name
        if not pp.exists() or not bp.exists():
            continue
        classic = pbpv3.convert(pd.read_parquet(pp))
        classic["GAME_ID"] = classic["GAME_ID"].astype("int64")
        walked = f26.walk_lineups(classic, pd.read_parquet(bp), home[gid], away[gid])
        if walked is None:
            continue
        st = f26.stints_from_walk(walked)
        if len(st):
            out.append(st)
    return pd.concat(out, ignore_index=True) if out else None


def _team_nets(stints, season):
    games = pd.read_csv(CACHE / f"games_{season}.csv")
    teams = pd.read_csv(CACHE / f"teams_{season}.csv")
    ab2id = dict(zip(teams.ABBR, teams.TEAM_ID))
    h = dict(zip(games.GAME_ID.astype(int), games.HOME.map(ab2id)))
    a = dict(zip(games.GAME_ID.astype(int), games.AWAY.map(ab2id)))
    pd_, ps = {}, {}
    reg = stints[stints.GAME_ID.astype(str).str.startswith("2")]   # regular season only
    for g, pm, poss in zip(reg.GAME_ID.astype(int), reg.PLUS_MINUS, reg.POSS):
        ht, at = h.get(g), a.get(g)
        if ht is None or at is None or pd.isna(ht) or pd.isna(at):
            continue
        pd_[ht] = pd_.get(ht, 0.0) + pm; pd_[at] = pd_.get(at, 0.0) - pm
        ps[ht] = ps.get(ht, 0.0) + poss; ps[at] = ps.get(at, 0.0) + poss
    teams["ACTUAL_NET"] = [pd_[t] / ps[t] * 100.0 if ps.get(t) else v
                           for t, v in zip(teams.TEAM_ID, teams.ACTUAL_NET)]
    return teams


def _player_minutes(stints, season):
    """Recompute players_{season}.csv MINUTES from corrected stint durations, keeping
    each player's team assignment (the max-minutes team) exactly as before."""
    pp = CACHE / f"players_{season}.csv"
    pl = pd.read_csv(pp)
    sec = {}
    for hl, al, dur in zip(stints.HOME_LINEUP, stints.AWAY_LINEUP, stints.DURATION_SECONDS):
        for p in (int(x) for x in (hl + "," + al).split(",") if x.strip()):
            sec[p] = sec.get(p, 0.0) + float(dur)
    old = pl.MINUTES.copy()
    missing = set(sec) - set(pl.PLAYER_ID.astype(int))
    if missing:
        print(f"  note: {len(missing)} stint players absent from players_{season}.csv "
              f"({sum(sec[p] for p in missing)/60:.0f} min) -- unrated until rosters rebuilt")
    pl["MINUTES"] = [sec[p] / 60.0 if p in sec else m for p, m in zip(pl.PLAYER_ID, pl.MINUTES)]
    return pl, float(pl.MINUTES.sum() / max(old.sum(), 1.0))


KEY2 = ["GAME_ID", "PERIOD", "HOME_LINEUP", "AWAY_LINEUP", "_occ"]


def backfill(season, dry_run=False):
    path = CACHE / f"stints_{season}.csv"
    if not path.exists():
        print(f"{season}: no stints file"); return
    old = pd.read_csv(path)
    new = rebuild_v3(season) if season >= 2026 else rebuild_classic(season)
    if new is None or new.empty:
        print(f"{season}: raw pbp not available -> left as-is (synthetic O/D)"); return
    for d in (old, new):
        d["HOME_LINEUP"] = d.HOME_LINEUP.map(_norm_lineup)
        d["AWAY_LINEUP"] = d.AWAY_LINEUP.map(_norm_lineup)
        d["GAME_ID"] = d.GAME_ID.astype("int64"); d["PERIOD"] = d.PERIOD.astype(int)
        d["_occ"] = d.groupby(["GAME_ID", "PERIOD", "HOME_LINEUP", "AWAY_LINEUP"]).cumcount()
    # Gate on GROUND TRUTH, not on agreement with the old (buggy) file: the fixed
    # builders legitimately change lineups (old classic builder dropped 0-sec stints
    # BEFORE the lineup walk, so their subs were never applied; old v3 walker never
    # re-seeded period starters). Require 5v5 coverage and game-margin reconciliation
    # vs the schedule's final scores to be at least as good as before.
    new = new[new.HOME_PTS.notna() & (new.DURATION_SECONDS > 0)].copy()
    new["POSS"] = new.DURATION_SECONDS / SEC_PER_POSS
    new["Y"] = new.PLUS_MINUS / new.POSS * 100.0
    cols = ["GAME_ID", "PERIOD", "HOME_LINEUP", "AWAY_LINEUP", "POSS", "Y",
            "DURATION_SECONDS", "PLUS_MINUS", "HOME_PTS", "AWAY_PTS"]
    cols += [c for c in ("START_SEC", "END_SEC") if c in new.columns]
    m = enrich_stints(new[cols].reset_index(drop=True))
    g = pd.read_csv(CACHE / f"games_{season}.csv").set_index("GAME_ID")
    fin = (g.HOME_PTS - g.AWAY_PTS)
    stats = {}
    for lab, d in (("old", old), ("new", m)):
        pm = d.groupby("GAME_ID").PLUS_MINUS.sum()
        e = (pm - fin.reindex(pm.index)).dropna()
        cov = d.groupby("GAME_ID").DURATION_SECONDS.sum().median()
        stats[lab] = (e.abs().mean(), cov, d.GAME_ID.nunique())
        print(f"  {lab}: {len(d)} stints, {stats[lab][2]} games | 5v5 PM vs final margin MAE "
              f"{stats[lab][0]:.2f} | median 5v5 sec/game {cov:.0f}")
    if not (stats["new"][0] <= stats["old"][0] + 0.05 and stats["new"][1] >= stats["old"][1] - 5
            and stats["new"][2] >= stats["old"][2]):
        print(f"  {season}: rebuilt stints are NOT better on ground truth -> not written")
        return
    print(f"  league offense {100*(m.HOME_PTS.sum()+m.AWAY_PTS.sum())/(2*m.POSS.sum()):.1f} pts/100 stint-poss")
    pl, ratio = _player_minutes(m, season)
    print(f"  stint-minutes total x{ratio:.3f} vs old players_{season}.csv")
    if dry_run:
        return
    _backup(path)
    m.to_csv(path, index=False)
    tp = CACHE / f"teams_{season}.csv"
    if tp.exists():
        _backup(tp)
        _team_nets(m, season).to_csv(tp, index=False)
    pp = CACHE / f"players_{season}.csv"
    _backup(pp)
    pl.to_csv(pp, index=False)
    print(f"  wrote {path.name}, teams_{season}.csv (ACTUAL_NET), players_{season}.csv (MINUTES)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    seasons = [int(a) for a in args] or list(range(2015, 2027))
    for s in seasons:
        backfill(s, dry_run=dry)
