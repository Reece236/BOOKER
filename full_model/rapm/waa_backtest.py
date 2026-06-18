"""
BOOKER WAA -- out-of-sample walk-forward backtest.

For each held-out season Y we estimate every player's on-court impact using ONLY
seasons < Y (box-prior-blended, multi-year, DARKO-decayed RAPM + aging), then
aggregate season-Y rosters/minutes into a predicted team net rating and convert
to wins with a net->wins map learned on the training seasons. Nothing from season
Y informs the player values, so this is a genuine forecast.

Outputs (full_model/rapm/):
  waa_backtest_team_predictions.csv   per team-season: actual vs predicted net & wins
  waa_backtest_metrics.csv            per-fold and pooled calibration metrics
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
ROOT = HERE.parent.parent
PLAYER_DATA = ROOT / "full_model" / "nba_player_data_2015-2025.csv"
TEAM_PRED = ROOT / "full_model" / "team_predictions.csv"

def cached_seasons():
    return sorted(int(p.stem.split("_")[1]) for p in CACHE.glob("stints_*.csv"))


SEASONS = cached_seasons()
TEST_SEASONS = [s for s in SEASONS if s >= 2018]   # need >=1 prior season of stints
N_PRIOR = 3                  # multi-year window
DECAY = 0.70                 # DARKO-style yearly decay
PRIOR_BASE = -1.0
PRIOR_K = 150.0
PRIOR_CLIP = (-12.0, 14.0)
AGE_PEAK = 27.0
AGE_QUAD = -0.03             # net-rating points lost per (yr from peak)^2
ALPHA_GRID = [1000, 2000, 4000, 8000]


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


def parse(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


# ---- load everything once ---------------------------------------------------
print("Loading cached stints + BOOKER box data ...")
STINTS, PLAYERS, TEAMS = {}, {}, {}
for s in SEASONS:
    d = pd.read_csv(CACHE / f"stints_{s}.csv")
    d["home"] = d.HOME_LINEUP.map(parse)
    d["away"] = d.AWAY_LINEUP.map(parse)
    STINTS[s] = d
    PLAYERS[s] = pd.read_csv(CACHE / f"players_{s}.csv")
    TEAMS[s] = pd.read_csv(CACHE / f"teams_{s}.csv")

bk = pd.read_csv(PLAYER_DATA)
bk["nm"] = bk.playerName.map(norm_name)
bk = bk.dropna(subset=["box"])
# box BPM and age per (name, season)
BOX = {(nm, ss): np.average(g.box, weights=g.minutesPlayed.clip(lower=1))
       for (nm, ss), g in bk.groupby(["nm", "season"])}
AGE = {(nm, ss): np.average(g.age, weights=g.minutesPlayed.clip(lower=1))
       for (nm, ss), g in bk.groupby(["nm", "season"])}

tp = pd.read_csv(TEAM_PRED)
ACTUAL_WINS = {(r.team_abbr, int(r.season)): r.actual_wins for r in tp.itertuples()}
OLD_PRED = {(r.team_abbr, int(r.season)): r.predicted_wins for r in tp.itertuples()}

# name per player id (latest season seen) for prior lookup + leaderboard
PID_NAME = {}
for s in SEASONS:
    for pid, nm in zip(PLAYERS[s].PLAYER_ID, PLAYERS[s].NAME):
        PID_NAME[pid] = nm

# actual wins for seasons absent from the legacy team_predictions file (e.g. 2026)
for s in SEASONS:
    gp = CACHE / f"games_{s}.csv"
    if not gp.exists():
        continue
    g = pd.read_csv(gp)
    reg = g[g.get("SEASON_TYPE", "Regular Season") == "Regular Season"]
    wins = {}
    for h, a, hw in zip(reg.HOME, reg.AWAY, reg.HOME_WIN):
        win = h if hw == 1 else a
        wins[win] = wins.get(win, 0) + 1
    for ab, w in wins.items():
        ACTUAL_WINS.setdefault((ab, s), w)


def build_impacts(train_seasons, target_season, alpha):
    """Box-prior-blended, decay-weighted ridge RAPM over train_seasons."""
    # decay-weighted total minutes per player (for prior shrinkage) and prior BPM
    wmin, wbpm, wsum = {}, {}, {}
    last_age = {}
    for s in train_seasons:
        w = DECAY ** (target_season - 1 - s)
        pl = PLAYERS[s]
        for pid, nm, mn in zip(pl.PLAYER_ID, pl.NAME, pl.MINUTES):
            key = norm_name(nm)
            bpm = BOX.get((key, s))
            if bpm is None:
                continue
            bpm = min(max(bpm, PRIOR_CLIP[0]), PRIOR_CLIP[1])
            wbpm[pid] = wbpm.get(pid, 0.0) + w * mn * bpm
            wmin[pid] = wmin.get(pid, 0.0) + w * mn
            wsum[pid] = wsum.get(pid, 0.0) + mn
            ag = AGE.get((key, s))
            if ag is not None:
                last_age[pid] = (s, ag)
    prior = {}
    for pid in wmin:
        raw = wbpm[pid] / wmin[pid] if wmin[pid] > 0 else PRIOR_BASE
        m = wsum.get(pid, 0.0)
        sh = m / (m + PRIOR_K)
        prior[pid] = raw * sh + PRIOR_BASE * (1 - sh)

    # stacked stint design
    rows, cols, vals, ys, ws = [], [], [], [], []
    all_ids = set()
    for s in train_seasons:
        for hl, al in zip(STINTS[s].home, STINTS[s].away):
            all_ids.update(hl); all_ids.update(al)
    all_ids = sorted(all_ids)
    col = {p: i for i, p in enumerate(all_ids)}
    ri = 0
    for s in train_seasons:
        d = STINTS[s]
        dw = DECAY ** (target_season - 1 - s)
        for hl, al, poss, y in zip(d.home, d.away, d.POSS, d.Y):
            for p in hl:
                rows.append(ri); cols.append(col[p]); vals.append(1.0)
            for p in al:
                rows.append(ri); cols.append(col[p]); vals.append(-1.0)
            ys.append(y); ws.append(poss * dw); ri += 1
    X = csr_matrix((vals, (rows, cols)), shape=(ri, len(all_ids)))
    y = np.array(ys); w = np.array(ws)
    b0 = np.array([prior.get(p, PRIOR_BASE) for p in all_ids])
    y_adj = y - X.dot(b0)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, y_adj, sample_weight=w)
    blended = b0 + ridge.coef_
    impact = dict(zip(all_ids, blended))
    return impact, prior, last_age


def aggregate_net(impact, season, last_age=None, target_season=None):
    """Predicted team net rating for `season` rosters using player impacts."""
    pl = PLAYERS[season]
    tmin = pl.groupby("TEAM_ID").MINUTES.sum().to_dict()
    pred = {}
    for pid, tid, mn in zip(pl.PLAYER_ID, pl.TEAM_ID, pl.MINUTES):
        if tid not in tmin or tmin[tid] <= 0:
            continue
        val = impact.get(pid, PRIOR_BASE)            # newcomers -> baseline
        if last_age is not None and target_season is not None and pid in last_age:
            s0, a0 = last_age[pid]
            a1 = a0 + (target_season - s0)
            val += AGE_QUAD * ((a1 - AGE_PEAK) ** 2 - (a0 - AGE_PEAK) ** 2)
        presence = mn / (tmin[tid] / 5.0)
        pred[tid] = pred.get(tid, 0.0) + val * presence
    return pred


def metrics(a, p):
    a, p = np.asarray(a), np.asarray(p)
    err = p - a
    return (np.sqrt(np.mean(err**2)), np.mean(np.abs(err)),
            np.corrcoef(a, p)[0, 1]**2, np.polyfit(p, a, 1)[0])


def main():
    abbr_of = {}
    for s in SEASONS:
        for tid, ab in zip(TEAMS[s].TEAM_ID, TEAMS[s].ABBR):
            abbr_of[tid] = ab

    rows = []
    fold_metrics = []
    for Y in TEST_SEASONS:
        train = [s for s in range(Y - N_PRIOR, Y) if s >= 2015]
        if not train:
            continue
        # pick alpha by predicting the most-recent training season's actual net
        val_season = max(train)
        best = None
        for alpha in ALPHA_GRID:
            impact, _, la = build_impacts(train, val_season, alpha)
            pred = aggregate_net(impact, val_season)
            act = dict(zip(TEAMS[val_season].TEAM_ID, TEAMS[val_season].ACTUAL_NET))
            ids = [t for t in pred if t in act]
            r2 = np.corrcoef([act[t] for t in ids], [pred[t] for t in ids])[0, 1] ** 2
            if best is None or r2 > best[1]:
                best = (alpha, r2)
        alpha = best[0]

        impact, prior, la = build_impacts(train, Y, alpha)

        # net->wins map learned on training seasons (actual net vs actual wins)
        tn, tw = [], []
        for s in train:
            for tid, net in zip(TEAMS[s].TEAM_ID, TEAMS[s].ACTUAL_NET):
                ab = abbr_of.get(tid)
                if (ab, s) in ACTUAL_WINS:
                    tn.append(net); tw.append(ACTUAL_WINS[(ab, s)])
        k, c = np.polyfit(tn, tw, 1)

        pred_net = aggregate_net(impact, Y, last_age=la, target_season=Y)
        act_net = dict(zip(TEAMS[Y].TEAM_ID, TEAMS[Y].ACTUAL_NET))
        a_net, p_net, a_win, p_win, old_win, abrs = [], [], [], [], [], []
        for tid in pred_net:
            ab = abbr_of.get(tid)
            if tid not in act_net or (ab, Y) not in ACTUAL_WINS:
                continue
            a_net.append(act_net[tid]); p_net.append(pred_net[tid])
            a_win.append(ACTUAL_WINS[(ab, Y)]); p_win.append(c + k * pred_net[tid])
            old_win.append(OLD_PRED.get((ab, Y), np.nan)); abrs.append(ab)
            rows.append({"season": Y, "team": ab,
                         "actual_net": round(act_net[tid], 2),
                         "pred_net": round(pred_net[tid], 2),
                         "actual_wins": ACTUAL_WINS[(ab, Y)],
                         "pred_wins": round(c + k * pred_net[tid], 1),
                         "old_model_wins": OLD_PRED.get((ab, Y), np.nan)})
        nrm = metrics(a_net, p_net)
        wrm = metrics(a_win, p_win)
        fold_metrics.append({"season": Y, "alpha": alpha, "train": ",".join(map(str, train)),
                             "net_rmse": round(nrm[0], 2), "net_r2": round(nrm[2], 3),
                             "wins_rmse": round(wrm[0], 2), "wins_mae": round(wrm[1], 2),
                             "wins_r2": round(wrm[2], 3), "wins_slope": round(wrm[3], 2),
                             "k_wins_per_net": round(k, 2)})
        print(f"Y={Y} alpha={alpha} train={train}  "
              f"netR2={nrm[2]:.3f} netRMSE={nrm[0]:.2f}  "
              f"winsRMSE={wrm[0]:.2f} winsR2={wrm[2]:.3f} slope={wrm[3]:.2f}")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(HERE / "waa_backtest_team_predictions.csv", index=False)

    # pooled metrics
    a_net = pred_df.actual_net.values; p_net = pred_df.pred_net.values
    a_win = pred_df.actual_wins.values; p_win = pred_df.pred_wins.values
    old = pred_df.old_model_wins.values
    pn = metrics(a_net, p_net); pw = metrics(a_win, p_win)
    pooled = {"season": "POOLED", "alpha": "", "train": f"{len(pred_df)} team-seasons",
              "net_rmse": round(pn[0], 2), "net_r2": round(pn[2], 3),
              "wins_rmse": round(pw[0], 2), "wins_mae": round(pw[1], 2),
              "wins_r2": round(pw[2], 3), "wins_slope": round(pw[3], 2), "k_wins_per_net": ""}
    fold_metrics.append(pooled)
    pd.DataFrame(fold_metrics).to_csv(HERE / "waa_backtest_metrics.csv", index=False)

    mask = ~np.isnan(old)
    obench = metrics(a_win[mask], old[mask]) if mask.sum() else (np.nan,)*4
    base_rmse = np.sqrt(np.mean((a_win - 41.0) ** 2))
    yr_lo, yr_hi = TEST_SEASONS[0], TEST_SEASONS[-1]
    print(f"\n=== POOLED OUT-OF-SAMPLE ({yr_lo - 1}-{str(yr_hi)[2:]}) ===")
    print(f"Net rating : RMSE={pn[0]:.2f}  R2={pn[2]:.3f}  slope={pn[3]:.2f}")
    print(f"Wins (WAA) : RMSE={pw[0]:.2f}  MAE={pw[1]:.2f}  R2={pw[2]:.3f}  slope={pw[3]:.2f}")
    print(f"Wins benchmark -- predict 41 for all : RMSE={base_rmse:.2f}")
    print(f"Wins benchmark -- old box model (retrodictive!) : RMSE={obench[0]:.2f}  R2={obench[2]:.3f}")
    print("\nWrote waa_backtest_team_predictions.csv and waa_backtest_metrics.csv")


if __name__ == "__main__":
    main()
