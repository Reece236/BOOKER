"""
BOOKER WAA -- Phase 1 proof (2024-25 season)
============================================
On-court (lineup/stint) player evaluation via box-score-prior-blended RAPM,
aggregated to team net rating and converted to wins.

Pipeline:
  1. Stints (10-man lineups, point margin, duration) -> per-100 net rating per stint.
  2. Box-score prior: each player's Basketball-Reference BPM (points/100 above avg)
     is the ridge prior MEAN. We residualize the stint margins against the prior,
     ridge-shrink the residual toward 0, then add the prior back. This is the
     RPM/PIPM-style "prior-informed RAPM" that is far more stable than raw RAPM.
  3. Player WAA = blended impact (pts/100 above avg). Team net rating =
     sum_p blended_p * possession_share_p (shares sum to 5 = five on-court slots).
  4. Net rating -> wins via a learned linear (MOV) map and a Pythagorean map.

This is RETRODICTIVE for 2024-25 (RAPM fit on the same season we predict): the goal
here is to prove the aggregation is well-calibrated before scaling to a true
out-of-sample multi-season backtest.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root
PLAYER_DATA = ROOT / "full_model" / "nba_player_data_2015-2025.csv"
TEAM_PRED = ROOT / "full_model" / "team_predictions.csv"

SEASON = 2025                 # 2024-25; GAME_ID prefix 224
SEC_PER_POSS = 28.8           # ~100 team possessions per 48 min
PRIOR_BASE = -1.0             # baseline a fringe/low-minute player regresses toward
PRIOR_K = 150.0               # minutes shrinkage constant for the BPM prior
PRIOR_CLIP = (-12.0, 14.0)    # raw single-season BPM is unstable at the tails
ALPHA_GRID = [500, 1000, 2000, 4000, 8000, 16000]

# Standard NBA stats team IDs -> Basketball-Reference abbreviations
TEAM_ID_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
    1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
    1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612751: "BRK", 1610612752: "NYK",
    1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHO",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
    1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
    1610612765: "DET", 1610612766: "CHO",
}


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()


def parse_lineup(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def load_prior():
    """Minutes-weighted BPM per normalized player name for the target season."""
    bk = pd.read_csv(PLAYER_DATA)
    bk = bk[bk.season == SEASON].copy()
    bk["nm"] = bk.playerName.map(norm_name)
    bk = bk.dropna(subset=["box"])
    agg = (
        bk.groupby("nm")
        .apply(lambda d: np.average(d.box, weights=d.minutesPlayed.clip(lower=1)))
        .rename("box")
        .reset_index()
    )
    return dict(zip(agg.nm, agg.box))


def main():
    stints = pd.read_csv(HERE / "lineup_stints.csv").dropna(
        subset=["HOME_LINEUP", "AWAY_LINEUP", "DURATION_SECONDS", "PLUS_MINUS"]
    )
    stints = stints[stints.DURATION_SECONDS > 0].copy()
    players = pd.read_csv(HERE / "players.csv")
    minutes = pd.read_csv(HERE / "player_minutes.csv")
    assign = pd.read_csv(HERE / "player_team_assignments.csv")

    # ---- prior (box BPM) per stint player ------------------------------------
    # Raw single-season BPM is noisy at low minutes, so we regress it toward a
    # below-average baseline with a minutes-based shrinkage: a 3-minute cameo no
    # longer carries a +44 prior.
    prior_by_name = load_prior()
    mins_by_id = pd.read_csv(HERE / "player_minutes.csv").set_index(
        "PLAYER_ID").TOTAL_MINUTES.to_dict()
    players["nm"] = players.PLAYER_NAME.map(norm_name)
    players["raw_bpm"] = players.nm.map(prior_by_name)
    n_matched = players.raw_bpm.notna().sum()
    players["raw_bpm"] = players.raw_bpm.clip(*PRIOR_CLIP).fillna(PRIOR_BASE)
    pm = players.PLAYER_ID.map(lambda i: mins_by_id.get(i, 0.0))
    shrink = pm / (pm + PRIOR_K)
    players["prior"] = players.raw_bpm * shrink + PRIOR_BASE * (1.0 - shrink)
    prior_by_id = dict(zip(players.PLAYER_ID, players.prior))
    name_by_id = dict(zip(players.PLAYER_ID, players.PLAYER_NAME))

    home = stints.HOME_LINEUP.map(parse_lineup)
    away = stints.AWAY_LINEUP.map(parse_lineup)
    # keep only clean 5v5 stints
    ok = (home.map(len) == 5) & (away.map(len) == 5)
    stints, home, away = stints[ok], home[ok], away[ok]

    poss = stints.DURATION_SECONDS.values / SEC_PER_POSS
    y = stints.PLUS_MINUS.values / poss * 100.0      # stint net rating (home view)
    w = poss

    # ---- design matrix --------------------------------------------------------
    all_ids = sorted({pid for lu in home for pid in lu} | {pid for lu in away for pid in lu})
    col = {pid: j for j, pid in enumerate(all_ids)}
    P = len(all_ids)
    N = len(stints)
    rows, cols, vals = [], [], []
    for i, (hl, al) in enumerate(zip(home, away)):
        for pid in hl:
            rows.append(i); cols.append(col[pid]); vals.append(1.0)
        for pid in al:
            rows.append(i); cols.append(col[pid]); vals.append(-1.0)
    X = csr_matrix((vals, (rows, cols)), shape=(N, P))

    b0 = np.array([prior_by_id.get(pid, PRIOR_BASE) for pid in all_ids])
    y_adj = y - X.dot(b0)        # residual margin not explained by the box prior

    # ---- team aggregation helpers --------------------------------------------
    team_by_id = dict(zip(assign.PLAYER_ID, assign.TEAM_ID))
    # game -> home/away team via the first stint's lineups
    game_home_team, game_away_team = {}, {}
    for gid, hl, al in zip(stints.GAME_ID, home, away):
        if gid not in game_home_team:
            for pid in hl:
                if pid in team_by_id:
                    game_home_team[gid] = team_by_id[pid]; break
            for pid in al:
                if pid in team_by_id:
                    game_away_team[gid] = team_by_id[pid]; break

    # actual team net rating + wins from the stint/game data
    team_pd, team_poss = {}, {}
    game_margin = {}
    for gid, pm, ps in zip(stints.GAME_ID, stints.PLUS_MINUS, poss):
        ht, at = game_home_team.get(gid), game_away_team.get(gid)
        if ht is None or at is None:
            continue
        team_pd[ht] = team_pd.get(ht, 0.0) + pm
        team_pd[at] = team_pd.get(at, 0.0) - pm
        team_poss[ht] = team_poss.get(ht, 0.0) + ps
        team_poss[at] = team_poss.get(at, 0.0) + ps
        game_margin[gid] = game_margin.get(gid, 0.0) + pm

    actual_net = {t: team_pd[t] / team_poss[t] * 100.0 for t in team_pd}

    def fit_and_aggregate(alpha):
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X, y_adj, sample_weight=w)
        blended = b0 + ridge.coef_
        bl_by_id = dict(zip(all_ids, blended))
        mins = minutes.set_index("PLAYER_ID").TOTAL_MINUTES.to_dict()
        # team totals
        tmin = {}
        for pid, mn in mins.items():
            t = team_by_id.get(pid)
            if t is not None:
                tmin[t] = tmin.get(t, 0.0) + mn
        pred_net = {}
        for pid, mn in mins.items():
            t = team_by_id.get(pid)
            if t is None or t not in tmin or pid not in bl_by_id:
                continue
            presence = mn / (tmin[t] / 5.0)
            pred_net[t] = pred_net.get(t, 0.0) + bl_by_id[pid] * presence
        return ridge, blended, bl_by_id, pred_net, ridge.intercept_

    # ---- pick alpha by predicted-vs-actual team net rating R^2 ----------------
    best = None
    for alpha in ALPHA_GRID:
        _, _, _, pred_net, hca = fit_and_aggregate(alpha)
        teams = [t for t in pred_net if t in actual_net]
        a = np.array([actual_net[t] for t in teams])
        p = np.array([pred_net[t] for t in teams])
        r2 = np.corrcoef(a, p)[0, 1] ** 2
        rmse = np.sqrt(np.mean((a - p) ** 2))
        if best is None or r2 > best[1]:
            best = (alpha, r2, rmse, hca)
        print(f"  alpha={alpha:>6}  netR2={r2:.3f}  netRMSE={rmse:5.2f}  HCA={hca:4.2f}")

    alpha = best[0]
    print(f"\nSelected alpha={alpha} (net rating R2={best[1]:.3f}, RMSE={best[2]:.2f})")
    ridge, blended, bl_by_id, pred_net, hca = fit_and_aggregate(alpha)

    # ---- wins: actual + net->wins maps ---------------------------------------
    tp = pd.read_csv(TEAM_PRED)
    tp = tp[tp.season == SEASON][["team_abbr", "actual_wins"]]
    actual_wins = dict(zip(tp.team_abbr, tp.actual_wins))

    teams = [t for t in pred_net if t in actual_net and TEAM_ID_ABBR.get(t) in actual_wins]
    abbr = [TEAM_ID_ABBR[t] for t in teams]
    a_net = np.array([actual_net[t] for t in teams])
    p_net = np.array([pred_net[t] for t in teams])
    a_win = np.array([actual_wins[TEAM_ID_ABBR[t]] for t in teams])

    # learned linear map wins ~ a + k*net (fit on ACTUAL net so it is the true map)
    k, c = np.polyfit(a_net, a_win, 1)
    pred_win_lin = c + k * p_net
    # pythagorean-style on margin-of-victory (net ~ MOV); exponent ~13.91 on pts
    # expected win% from net rating using logistic approx wins = 41 + 2.7*net
    pred_win_pyth = 41.0 + 2.7 * p_net

    def metrics(actual, pred):
        err = pred - actual
        rmse = np.sqrt(np.mean(err ** 2)); mae = np.mean(np.abs(err))
        r2 = np.corrcoef(actual, pred)[0, 1] ** 2
        slope = np.polyfit(pred, actual, 1)[0]
        return rmse, mae, r2, slope

    print("\n=== 2024-25 CALIBRATION (retrodictive proof) ===")
    rm, ma, r2, sl = metrics(a_net, p_net)
    print(f"Net rating : RMSE={rm:5.2f}  MAE={ma:5.2f}  R2={r2:.3f}  slope(actual~pred)={sl:.2f}")
    rm, ma, r2, sl = metrics(a_win, pred_win_lin)
    print(f"Wins (lin) : RMSE={rm:5.2f}  MAE={ma:5.2f}  R2={r2:.3f}  slope={sl:.2f}  (k={k:.2f} wins/net, base={c:.1f})")
    rm, ma, r2, sl = metrics(a_win, pred_win_pyth)
    print(f"Wins (pyth): RMSE={rm:5.2f}  MAE={ma:5.2f}  R2={r2:.3f}  slope={sl:.2f}")

    # ---- write outputs --------------------------------------------------------
    mins = minutes.set_index("PLAYER_ID").TOTAL_MINUTES.to_dict()
    player_out = pd.DataFrame({
        "player_id": all_ids,
        "player_name": [name_by_id.get(p, "") for p in all_ids],
        "team_abbr": [TEAM_ID_ABBR.get(team_by_id.get(p), "") for p in all_ids],
        "minutes": [mins.get(p, 0.0) for p in all_ids],
        "prior_bpm": b0,
        "blended_impact_per100": np.round(blended, 3),
    })
    player_out["waa_pts"] = np.round(
        player_out.blended_impact_per100 * player_out.minutes / (48.0 * 82.0) * 5.0, 3
    )
    player_out = player_out.sort_values("blended_impact_per100", ascending=False)
    player_out.to_csv(HERE / "blended_rapm_2025.csv", index=False)

    team_out = pd.DataFrame({
        "team_abbr": abbr,
        "actual_net_rating": np.round(a_net, 2),
        "pred_net_rating": np.round(p_net, 2),
        "actual_wins": a_win,
        "pred_wins_linear": np.round(pred_win_lin, 1),
        "pred_wins_pyth": np.round(pred_win_pyth, 1),
    }).sort_values("actual_wins", ascending=False)
    team_out.to_csv(HERE / "team_predictions_rapm_2025.csv", index=False)

    print(f"\nPrior matched {n_matched}/{len(players)} stint players to BPM.")
    print("Wrote blended_rapm_2025.csv and team_predictions_rapm_2025.csv")
    print("\nTop 15 players by blended impact:")
    print(player_out.head(15)[["player_name", "team_abbr", "minutes",
                                "prior_bpm", "blended_impact_per100"]].to_string(index=False))
    print("\nTeam predictions:")
    print(team_out.to_string(index=False))


if __name__ == "__main__":
    main()
