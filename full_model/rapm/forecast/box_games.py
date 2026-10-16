"""
Per-player PER-GAME box lines from play-by-play -> cache/box_games.parquet.

Both feeds:
  classic nbastats_{s}.csv (2015-25): EVENTMSGTYPE 1 make (P1 shooter, P2 assister),
      2 miss (P3 blocker), 3 FT, 4 rebound, 5 turnover (P2 stealer if "STEAL"), 6 foul
  v3 parquet pbp_{s}/ (2026): actionType rows; assister parsed from "(Name n AST)",
      steals/blocks from their own "X STEAL (n STL)" / "X BLOCK (n BLK)" rows
Rebounds split OREB/DREB by the team of the last missed shot / free throw.
On-court possessions per player-game come from the regular-season stints (team-average
possessions of every stint he is on the floor for) -- the denominator for per-100 rates.

Columns: season, GAME_ID, PLAYER_ID, poss, pts, fga, fg3a, fg3m, fta, ftm, ast, oreb,
         dreb, stl, blk, tov, pf
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
OUT = CACHE / "box_games.parquet"
STATS = ["pts", "fga", "fg3a", "fg3m", "fta", "ftm", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf"]


def _classic(s):
    d = pd.read_csv(CACHE / f"nbastats_{s}.csv", low_memory=False,
                    usecols=["GAME_ID", "EVENTNUM", "PERIOD", "EVENTMSGTYPE", "PLAYER1_ID", "PLAYER1_TEAM_ID",
                             "PLAYER2_ID", "PLAYER3_ID", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"])
    d = d[d.GAME_ID.astype(str).str.lstrip("0").str.startswith("2")]
    d = d.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)
    txt = (d.HOMEDESCRIPTION.fillna("") + " " + d.VISITORDESCRIPTION.fillna("") + " " + d.NEUTRALDESCRIPTION.fillna("")).str.upper()
    t = d.EVENTMSGTYPE
    p1 = d.PLAYER1_ID.fillna(0).astype("int64"); p2 = d.PLAYER2_ID.fillna(0).astype("int64"); p3 = d.PLAYER3_ID.fillna(0).astype("int64")
    is3 = txt.str.contains("3PT", regex=False)
    ftmiss = (t == 3) & txt.str.contains("MISS", regex=False)
    team = d.PLAYER1_TEAM_ID
    miss_team = team.where((t == 2) | ftmiss).groupby([d.GAME_ID, d.PERIOD]).ffill()
    rows = []
    def add(mask, pid, col, val=1.0):
        m = mask & (pid > 0) & (pid < 1610612737)
        rows.append(pd.DataFrame({"GAME_ID": d.GAME_ID[m].astype("int64"), "PLAYER_ID": pid[m], "stat": col, "v": val if np.isscalar(val) else val[m]}))
    add(t == 1, p1, "pts", np.where(is3, 3.0, 2.0)); add(t.isin([1, 2]), p1, "fga")
    add(t.isin([1, 2]) & is3, p1, "fg3a"); add((t == 1) & is3, p1, "fg3m")
    add(t == 3, p1, "fta"); add((t == 3) & ~ftmiss, p1, "ftm"); add((t == 3) & ~ftmiss, p1, "pts")
    add(t == 1, p2, "ast"); add(t == 2, p3, "blk")
    add((t == 5) & txt.str.contains("STEAL", regex=False), p2, "stl"); add(t == 5, p1, "tov"); add(t == 6, p1, "pf")
    reb = (t == 4)
    add(reb & (team == miss_team), p1, "oreb"); add(reb & (team != miss_team), p1, "dreb")
    return pd.concat(rows, ignore_index=True)


_AST = re.compile(r"\(([^()]+?) \d+ AST\)")      # any script: Jokić, Dončić, Šarić
_STL = re.compile(r"STEAL \(\d+ STL\)")
_BLK = re.compile(r"BLOCK \(\d+ BLK\)")


def _fold(s):
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _v3(s):
    rows = []
    for f in sorted(glob.glob(str(CACHE / f"pbp_{s}" / "*.parquet"))):
        d = pd.read_parquet(f)
        if not len(d) or not str(d.gameId.iloc[0]).lstrip("0").startswith("2"):
            continue
        gid = int(d.gameId.iloc[0])
        d = d.sort_values(["period", "actionNumber"]).reset_index(drop=True)
        pid = pd.to_numeric(d.personId, errors="coerce").fillna(0).astype("int64")
        team = pd.to_numeric(d.teamId, errors="coerce")
        at, desc = d.actionType.fillna(""), d.description.fillna("")
        sv = pd.to_numeric(d.shotValue, errors="coerce").fillna(0)
        ftmiss = (at == "Free Throw") & ((d.shotResult == "Missed") | desc.str.upper().str.contains("MISS"))
        miss_team = team.where((at == "Missed Shot") | ftmiss).groupby(d.period).ffill()
        roster = {}
        for p_, nm in zip(pid, d.playerName):
            if p_ > 0 and isinstance(nm, str) and nm:
                roster.setdefault(nm, int(p_))
                roster.setdefault(_fold(nm), int(p_))     # descriptions are ASCII ("Jokic")
        acc = []
        for i in range(len(d)):
            p_, a = int(pid.iat[i]), at.iat[i]
            if p_ <= 0:
                continue
            if a in ("Made Shot", "Missed Shot"):
                acc.append((p_, "fga", 1.0))
                if sv.iat[i] == 3:
                    acc.append((p_, "fg3a", 1.0))
                if a == "Made Shot":
                    acc.append((p_, "pts", float(sv.iat[i] or 2)))
                    if sv.iat[i] == 3:
                        acc.append((p_, "fg3m", 1.0))
                    m = _AST.search(desc.iat[i])
                    if m:
                        nm = m.group(1).strip()
                        ap = roster.get(nm) or roster.get(_fold(nm)) or roster.get(nm.split()[-1])
                        if ap:
                            acc.append((ap, "ast", 1.0))
            elif a == "Free Throw":
                acc.append((p_, "fta", 1.0))
                if not ftmiss.iat[i]:
                    acc += [(p_, "ftm", 1.0), (p_, "pts", 1.0)]
            elif a == "Rebound":
                acc.append((p_, "oreb" if team.iat[i] == miss_team.iat[i] else "dreb", 1.0))
            elif a == "Turnover":
                acc.append((p_, "tov", 1.0))
            elif a == "Foul":
                acc.append((p_, "pf", 1.0))
            elif _STL.search(desc.iat[i]):
                acc.append((p_, "stl", 1.0))
            elif _BLK.search(desc.iat[i]):
                acc.append((p_, "blk", 1.0))
        if acc:
            x = pd.DataFrame(acc, columns=["PLAYER_ID", "stat", "v"]); x["GAME_ID"] = gid
            rows.append(x)
    return pd.concat(rows, ignore_index=True)


def player_game_poss(data, s):
    st = data.STINTS.get(s)
    st = st[st.GAME_ID.astype(str).str.startswith("2")]
    acc = {}
    for gid, hl, al, poss in zip(st.GAME_ID.astype("int64"), st.home, st.away, st.POSS):
        for p in list(hl) + list(al):
            acc[(gid, p)] = acc.get((gid, p), 0.0) + float(poss)
    return pd.DataFrame([(g, p, v) for (g, p), v in acc.items()], columns=["GAME_ID", "PLAYER_ID", "poss"])


def build(seasons=range(2015, 2027)):
    from . import player_impacts as pi
    data = pi.BookerData(seasons)
    out = []
    for s in seasons:
        if (CACHE / f"nbastats_{s}.csv").exists():
            ev = _classic(s)
        elif glob.glob(str(CACHE / f"pbp_{s}" / "*.parquet")):
            ev = _v3(s)
        else:
            continue
        w = ev.pivot_table(index=["GAME_ID", "PLAYER_ID"], columns="stat", values="v", aggfunc="sum", fill_value=0.0)
        w = w.reindex(columns=STATS, fill_value=0.0).reset_index()
        w = w.merge(player_game_poss(data, s), on=["GAME_ID", "PLAYER_ID"], how="inner")
        w.insert(0, "season", s)
        out.append(w)
        print(f"  {s}: {len(w)} player-games, {w.GAME_ID.nunique()} games, pts/100 {100*w.pts.sum()/w.poss.sum():.1f}", flush=True)
    B = pd.concat(out, ignore_index=True)
    B.to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(B)} rows)")
    return B


if __name__ == "__main__":
    build()


# ------------------------------------------------------------------ box -> impact map
FEATS = ["pts", "fga", "fta", "fg3m", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf"]
MIN_POSS_FIT = 1000      # ~500 minutes


def rates(df):
    return df[FEATS].div(df.poss.clip(lower=1.0), axis=0) * 100.0


def fit_box_map(B, rat, seasons, alpha=50.0):
    """BPM-style ridge: season per-100 box rates -> BookerFormer impact_total (rating row of
    the same season), poss-weighted, on `seasons` only. Returns (model, tau2, sigma2):
      tau2   = variance of the persistent player-level misfit (impact - f(season box)),
               net of the rating's own posterior variance
      sigma2 = per-possession game-to-game variance of f(game box): Var(y_g) ~ sigma2/poss_g
    """
    from sklearn.linear_model import Ridge
    b = B[B.season.isin(seasons)]
    agg = b.groupby(["season", "PLAYER_ID"])[FEATS + ["poss"]].sum().reset_index()
    agg = agg.merge(rat[["season", "PLAYER_ID", "impact_total", "sd_off", "sd_def"]], on=["season", "PLAYER_ID"])
    agg = agg[agg.poss >= MIN_POSS_FIT]
    X, y, w = rates(agg), agg.impact_total, agg.poss
    m = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    res = y - m.predict(X)
    tau2 = max(float(np.average(res ** 2, weights=w) - np.average(agg.sd_off ** 2 + agg.sd_def ** 2, weights=w)), 0.5)
    g = b[b.poss >= 5].copy()
    g["yb"] = m.predict(rates(g))
    gm = g.groupby(["season", "PLAYER_ID"]).apply(lambda x: np.average(x.yb, weights=x.poss)).rename("ybar")
    g = g.join(gm, on=["season", "PLAYER_ID"])
    sigma2 = float(np.mean(g.poss * (g.yb - g.ybar) ** 2))
    return m, tau2, sigma2
