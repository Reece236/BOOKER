"""Availability + minutes-redistribution research (inputs for forecast/sequential.py).

(a) P(plays tonight | missed his team's last k games): the honest pre-tip availability
    signal that exists historically (long absences are public before tip).
(b) Redistribution: when players sit, where do their minutes go? Fit on REALIZED per-game
    presence given who actually played:
        pred_q = c_q + M * w_q / sum(w),  w_q = c_q^alpha * (1 + gamma * S_q)
    c_q = player's recent share-when-playing, M = minutes-share of absent rotation
    players, S_q = fraction of the absent minutes at q's position group. alpha < 1 =
    deeper bench absorbs relatively more; gamma > 0 = same-position backups absorb more.
    Proportional (current) = alpha 1, gamma 0. Fit 2019-21, test 2022-26.
Run: python availability_research.py
"""
import numpy as np
import pandas as pd

from forecast import minutes_share as ms

SEASONS = range(2019, 2027)
HL = 10.0


def season_games(ctx, s):
    """Per team-game records: candidates with recent share-when-playing, availability,
    realized presence, missed-streak before the game, position group."""
    tgs = ctx.tgs[s]
    tsec = tgs.groupby(["GAME_ID", "team"]).sec.transform("sum")
    tgs = tgs.assign(pres=tgs.sec / (tsec / 5.0))
    order = tgs[["GAME_ID", "DATE", "team"]].drop_duplicates().sort_values(["team", "DATE", "GAME_ID"])
    rows = []
    for team, gs in order.groupby("team"):
        hist = []                       # list of {pid: pres} (0 for DNP on roster)
        roster = set()
        for gid in gs.GAME_ID:
            act = dict(tgs[(tgs.GAME_ID == gid) & (tgs.team == team)][["PLAYER_ID", "pres"]].values)
            act = {int(k): v for k, v in act.items()}
            if len(hist) >= 5:
                n = len(hist)
                w = 0.5 ** ((n - 1 - np.arange(n)) / HL)
                for p in roster | set(act):
                    played = [(wk, g[p]) for wk, g in zip(w, hist) if g.get(p, 0) > 0]
                    if not played and p not in act:
                        continue
                    c = (sum(wk * x for wk, x in played) / sum(wk for wk, _ in played)) if played else np.nan
                    streak = 0
                    for g in reversed(hist):
                        if g.get(p, 0) > 0:
                            break
                        streak += 1
                    if not played:
                        streak = -1     # never played for this team yet (new arrival)
                    rows.append((s, gid, team, p, c, int(p in act), act.get(p, 0.0), streak,
                                 ctx.position(p, s)))
            roster |= set(act)
            hist.append({p: act.get(p, 0.0) for p in roster})
    return pd.DataFrame(rows, columns=["season", "GAME_ID", "team", "PLAYER_ID", "c", "played",
                                       "pres", "streak", "pos"])


def redistribute(g, alpha, gamma, avail_col="played"):
    a = g[avail_col].to_numpy(float)
    c = np.nan_to_num(g.c.to_numpy(float), nan=0.05)
    M = np.sum((1 - a) * c)
    absent_pos = {}
    for pos, ci, ai in zip(g.pos, c, a):
        absent_pos[pos] = absent_pos.get(pos, 0.0) + (1 - ai) * ci
    S = np.array([absent_pos.get(p, 0.0) / M if M > 0 else 0.0 for p in g.pos])
    w = a * c ** alpha * (1 + gamma * S)
    pred = a * c + (M * w / w.sum() if w.sum() > 0 else 0)
    return pred * 5.0 / pred.sum() if pred.sum() > 0 else pred


def main():
    ctx = ms._Ctx()
    G = pd.concat([season_games(ctx, s) for s in SEASONS], ignore_index=True)
    print(f"{G.GAME_ID.nunique():,} games, {len(G):,} player-game candidates")
    # (a) availability
    t = G[G.streak >= 0].copy()
    t["k"] = pd.cut(t.streak, [-1, 0, 1, 2, 4, 9, 999], labels=["0 (played last)", "1", "2", "3-4", "5-9", "10+"])
    print("\n(a) P(plays tonight | consecutive team games just missed)")
    print(t.groupby("k", observed=True).played.agg(["mean", "size"]).round(3).T.to_string())
    # (b) redistribution
    fit = G[G.season <= 2021]; test = G[G.season >= 2022]
    def err(D, a, gm):
        e = []
        for _, g in D.groupby(["GAME_ID", "team"]):
            if g.played.sum() < 5:
                continue
            pr = redistribute(g, a, gm)
            m = g.played.to_numpy() == 1
            e.append(((pr - g.pres.to_numpy())[m]) ** 2)
        return np.sqrt(np.mean(np.concatenate(e)))
    fit_s = fit[fit.GAME_ID.isin(fit.GAME_ID.drop_duplicates().sample(3000, random_state=0))]
    best = min(((a, gm, err(fit_s, a, gm)) for a in (0.4, 0.6, 0.8, 1.0, 1.2) for gm in (0, 1, 2, 4)),
               key=lambda x: x[2])
    print(f"\n(b) redistribution fit (2019-21 sample): alpha {best[0]}, gamma {best[1]}, RMSE {best[2]:.4f}")
    for lab, a, gm in (("proportional (current)", 1.0, 0.0), ("fitted", best[0], best[1])):
        print(f"  test 2022-26 presence RMSE, {lab:24s}: {err(test, a, gm):.4f}")
    G.to_parquet(ms.CACHE / "availability_panel.parquet")


if __name__ == "__main__":
    main()
