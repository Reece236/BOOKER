"""
Player-metric evaluation harness: BOOKER vs PyMC-Bayes vs enhanced-ridge vs pure
single-season RAPM vs box (VORP-rate) [+ DARKO / LEBRON if fetched].

Evaluations
  1. STICKINESS  : age-adjusted within-player year-over-year correlation
  2. SIMILARITY  : same-season cross-metric Spearman matrix
  3. PERFORMANCE : metric_t predicting t+1 ground truths -- pure RAPM t+1 (context-
                   true, noisy), real on-court net t+1, and team-level net RMSE
  4. TEAM-CHANGE : stickiness/prediction split movers vs stayers (context robustness)
  5. UNIQUE READS: biggest BOOKER-vs-X disagreements; whom does t+1 vindicate?
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

RAPM = Path(__file__).resolve().parent
sys.path.insert(0, str(RAPM))
CACHE = RAPM / "cache"
# External metric histories live in cache/ (they used to be read from an ephemeral
# agent scratchpad under /private/tmp -- once it was cleared, DARKO/LEBRON were
# silently skipped and the "vs DARKO" scoreline could not be reproduced).
SCRATCH = CACHE
from forecast import player_impacts as pi
from forecast import uncertainty as unc

MIN_MIN = 1000


def build_panel():
    bk = pd.read_csv(RAPM / "booker_bookerformer_ratings.csv")
    panel = bk[["PLAYER_ID", "season", "player", "team", "minutes"]].copy()
    panel["BOOKER"] = bk.impact_total
    by = pd.read_csv(RAPM / "booker_bayesian_ratings.csv")
    if "impact_total" in by.columns:
        panel = panel.merge(by[["PLAYER_ID", "season", "impact_total"]].rename(
            columns={"impact_total": "BAYES"}), on=["PLAYER_ID", "season"], how="left")
    en = pd.read_csv(RAPM / "booker_waa_enhanced_ratings.csv")
    if "pid" in en.columns:
        en = en.rename(columns={"pid": "PLAYER_ID"})
    if "impact_total" in en.columns:
        panel = panel.merge(en[["PLAYER_ID", "season", "impact_total"]].rename(
            columns={"impact_total": "ENH_RIDGE"}), on=["PLAYER_ID", "season"], how="left")
    pr = pd.read_csv(CACHE / "pure_rapm.csv")
    panel = panel.merge(pr[["PLAYER_ID", "season", "pure_rapm"]].rename(
        columns={"pure_rapm": "PURE_RAPM"}), on=["PLAYER_ID", "season"], how="left")
    # box VORP rate from master (join on name-season)
    m = pd.read_csv(pi.ROOT / "full_model" / "nba_master_dataset_with_archetypes.csv",
                    usecols=["playerName", "season", "vorp", "minutesPlayed"])
    m["nm"] = m.playerName.map(pi.norm_name)
    m["season"] = pd.to_numeric(m.season, errors="coerce")
    m = m.dropna(subset=["season"]); m["season"] = m.season.astype(int)
    m = m.sort_values("minutesPlayed").drop_duplicates(["nm", "season"], keep="last")
    m["BOX_VORP"] = 1500.0 * pd.to_numeric(m.vorp, errors="coerce") / m.minutesPlayed.clip(lower=1)
    panel["nm"] = panel.player.map(pi.norm_name)
    panel = panel.merge(m[["nm", "season", "BOX_VORP"]], on=["nm", "season"], how="left")
    # external metrics if the fetch agent delivered
    for name, f in (("DARKO", "darko_history.csv"), ("LEBRON", "lebron_history.csv")):
        p = SCRATCH / f
        if not p.exists():
            print(f"WARNING: {p} missing -> {name} NOT evaluated")
        if p.exists():
            try:
                e = pd.read_csv(p)
                e["nm"] = e.player.map(pi.norm_name)
                e = e[["nm", "season", "metric"]].rename(columns={"metric": name})
                e["season"] = e.season.astype(int)
                panel = panel.merge(e.drop_duplicates(["nm", "season"]), on=["nm", "season"], how="left")
                print(f"joined {name}: {panel[name].notna().sum()} rows matched")
            except Exception as exc:
                print(f"({name} join skipped: {exc})")
    # ground truths
    lc = pd.read_csv(CACHE / "lineup_context.csv")[["PLAYER_ID", "season", "real_pm"]]
    panel = panel.merge(lc, on=["PLAYER_ID", "season"], how="left")
    a = unc.attach_attrs(panel[["player", "season"]].assign(minutes=panel.minutes))
    panel["age"] = a.age.values
    return panel


def age_resid(df, col):
    d = df.dropna(subset=[col, "age"])
    A = np.column_stack([np.ones(len(d)), d.age, d.age ** 2])
    b, *_ = np.linalg.lstsq(A, d[col], rcond=None)
    r = pd.Series(np.nan, index=df.index)
    r.loc[d.index] = d[col] - A @ b
    return r


def main():
    panel = build_panel()
    METRICS = [c for c in ("BOOKER", "BAYES", "ENH_RIDGE", "PURE_RAPM", "BOX_VORP",
                           "DARKO", "LEBRON") if c in panel.columns and panel[c].notna().sum() > 500]
    print("metrics in panel:", METRICS, f"| rows {len(panel)}")

    q = panel[panel.minutes >= MIN_MIN].copy()
    nxt = panel.rename(columns={c: c + "_n" for c in METRICS + ["real_pm", "team", "minutes"]})
    nxt = nxt.copy(); nxt["season"] -= 1
    keep_n = ["PLAYER_ID", "season"] + [c + "_n" for c in METRICS + ["real_pm", "team", "minutes"]]
    pair = q.merge(nxt[keep_n], on=["PLAYER_ID", "season"], how="inner")
    pair = pair[pair.minutes_n >= MIN_MIN].copy()
    pair["moved"] = (pair.team != pair.team_n).astype(int)
    print(f"y2y pairs (>= {MIN_MIN} min both): {len(pair)} | movers {pair.moved.sum()}")

    # 1) age-adjusted stickiness (+ mover split)
    print("\n1) STICKINESS (age-adjusted y2y corr)   all | stayers | movers | mover penalty")
    for mcol in METRICS:
        d = pair.dropna(subset=[mcol, mcol + "_n", "age"])
        if len(d) < 200: print(f"  {mcol:<10} (n too small)"); continue
        r0 = age_resid(d, mcol); r1 = age_resid(d.rename(columns={mcol + "_n": "zz", "age": "age"}).assign(age=d.age + 1), None) if False else None
        # residualize both years on age (t uses age, t+1 uses age+1)
        d = d.assign(_r0=age_resid(d, mcol).values,
                     _r1=age_resid(d.assign(**{mcol: d[mcol + "_n"], "age": d.age + 1}), mcol).values)
        def cc(dd): return spearmanr(dd._r0, dd._r1)[0] if len(dd) > 30 else np.nan
        call, cst, cmv = cc(d), cc(d[d.moved == 0]), cc(d[d.moved == 1])
        print(f"  {mcol:<10} {call:.3f} | {cst:.3f} | {cmv:.3f} | {cst - cmv:+.3f}   (n={len(d)}, movers={int(d.moved.sum())})")

    # 2) similarity matrix (same season)
    print("\n2) SIMILARITY (same-season Spearman, >=1000 min)")
    hdr = "         " + "".join(f"{m[:8]:>10}" for m in METRICS); print(hdr)
    for a_ in METRICS:
        row = f"  {a_[:7]:<7}"
        for b_ in METRICS:
            d = q.dropna(subset=[a_, b_])
            row += f"{spearmanr(d[a_], d[b_])[0]:>10.3f}" if len(d) > 100 else f"{'—':>10}"
        print(row)

    # 3) forward performance
    print("\n3) PERFORMANCE: metric_t -> t+1 truth (Spearman)   pureRAPM_t+1 | real_pm_t+1 | movers-only pureRAPM")
    for mcol in METRICS:
        d = pair.dropna(subset=[mcol, "PURE_RAPM_n"])
        c1 = spearmanr(d[mcol], d.PURE_RAPM_n)[0] if len(d) > 100 else np.nan
        d2 = pair.dropna(subset=[mcol, "real_pm_n"])
        c2 = spearmanr(d2[mcol], d2.real_pm_n)[0] if len(d2) > 100 else np.nan
        dm = d[d.moved == 1]
        c3 = spearmanr(dm[mcol], dm.PURE_RAPM_n)[0] if len(dm) > 60 else np.nan
        print(f"  {mcol:<10} {c1:.3f} | {c2:.3f} | {c3:.3f}")

    # 3b) team-level: minutes-weighted metric -> t+1 team net RMSE
    print("\n3b) TEAM net t+1 RMSE (roll up metric_t on t+1 rosters/minutes)")
    tn = {}
    data = pi.BookerData()
    for s in range(2019, 2027):
        if s not in data.TEAMS: continue
        tn[s] = dict(zip(data.TEAMS[s].TEAM_ID, data.TEAMS[s].ACTUAL_NET))
    pl = {s: data.PLAYERS[s] for s in range(2019, 2027) if s in data.PLAYERS}
    for mcol in METRICS:
        errs = []
        look = panel.dropna(subset=[mcol]).set_index(["PLAYER_ID", "season"])[mcol].to_dict()
        for s in range(2019, 2027):
            if s not in pl: continue
            P = pl[s]; tmin = {}
            for pid, tid, mn in zip(P.PLAYER_ID, P.TEAM_ID, P.MINUTES):
                tmin[tid] = tmin.get(tid, 0.0) + mn
            pred = {}
            for pid, tid, mn in zip(P.PLAYER_ID, P.TEAM_ID, P.MINUTES):
                v = look.get((int(pid), s - 1))
                if v is None: v = -1.0
                pred[tid] = pred.get(tid, 0.0) + v * (mn / (tmin[tid] / 5.0))
            for tid, pv in pred.items():
                act = tn.get(s, {}).get(tid)
                if act is not None and act == act:
                    errs.append(pv - act)
        errs = np.array(errs)
        # allow intercept+scale correction so different metric scales compare fairly
        # (regress act on pred is cheating; instead standardize: corr-based R2)
        print(f"  {mcol:<10} raw RMSE {np.sqrt((errs**2).mean()):6.2f}  (n={len(errs)})")

    # 5) unique reads: BOOKER vs each — who wins the disagreements?
    print("\n5) UNIQUE READS: top-decile |BOOKER z - X z| disagreements; t+1 pureRAPM sides with…")
    zq = lambda s: (s - s.mean()) / s.std()
    for other in [m for m in METRICS if m != "BOOKER"]:
        d = pair.dropna(subset=["BOOKER", other, "PURE_RAPM_n"]).copy()
        d["zb"], d["zo"] = zq(d.BOOKER), zq(d[other])
        d["gap"] = d.zb - d.zo
        top = d[d.gap.abs() >= d.gap.abs().quantile(0.9)]
        zt = zq(d.PURE_RAPM_n)
        wins = ((top.zb - top.zo) * (zt.loc[top.index] - (top.zb + top.zo) / 2) > 0)
        print(f"  vs {other:<10}: {len(top)} disagreements, BOOKER side vindicated {100*wins.mean():.0f}%")
        ex = top.reindex(top.gap.abs().sort_values(ascending=False).index)[:4]
        for _, x in ex.iterrows():
            print(f"     {x.player:24s} {int(x.season)}: BOOKER z {x.zb:+.1f} vs {other} z {x.zo:+.1f} -> t+1 RAPM {x.PURE_RAPM_n:+.1f}")


if __name__ == "__main__":
    main()
