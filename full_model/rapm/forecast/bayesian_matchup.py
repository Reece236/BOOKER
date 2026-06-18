"""
Full Bayesian stint attribution (PyMC): offensive & defensive pts/100 per matchup.

Each stint supplies:
  Y_OFF_HOME — home team points scored per 100 possessions
  Y_DEF_HOME — away team points scored per 100 possessions (home defense)

The model allocates credit across all 10 players with:
  * hierarchical player offensive & defensive effects (box-score priors)
  * sparse same-team offensive synergy (teammate fit)
  * sparse cross-team matchup terms (away offense vs home defense)

Fits with ADVI for speed, then draws posterior samples for player distributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import sparse

from . import player_impacts as pi

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
OUT = CACHE / "bayesian"

MAX_STINTS = 14000
MAX_PAIR = 350
MAX_MATCH = 250
ADVI_STEPS = 25_000
POST_SAMPLES = 800

SUBSAMPLE_SEED = 42


def parse_lineup(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def _pair_key(a, b):
    return (a, b) if a < b else (b, a)


def load_stints(data, seasons):
    parts = []
    for s in seasons:
        d = data.STINTS.get(s)
        if d is None:
            continue
        t = d.copy()
        if "Y_OFF_HOME" not in t.columns:
            import sys
            sys.path.insert(0, str(RAPM))
            from stint_off_def import enrich_stints  # noqa: WPS433
            t = enrich_stints(t)
        t["season"] = s
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def top_pairs(stints, side="team", max_pairs=MAX_PAIR):
    """Count co-possessions for same-team (off synergy) or cross-team (matchup) pairs."""
    co = {}
    home = stints.home if hasattr(stints, "home") else stints["home"]
    away = stints.away if hasattr(stints, "away") else stints["away"]
    poss = stints.POSS if hasattr(stints, "POSS") else stints["POSS"]
    for hl, al, po in zip(home, away, poss):
        if side == "team":
            for lst in (hl, al):
                ids = sorted(lst)
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        k = _pair_key(ids[i], ids[j])
                        co[k] = co.get(k, 0.0) + po
        else:
            for i in hl:
                for j in al:
                    k = (i, j)
                    co[k] = co.get(k, 0.0) + po
    items = sorted(co.items(), key=lambda kv: -kv[1])[:max_pairs]
    return [k for k, _ in items]


def build_design(stints, players, pair_idx, match_idx):
    """Sparse design for off and def stint regressions."""
    n = len(stints)
    p = len(players)
    pid_to_i = {pid: i for i, pid in enumerate(players)}

    def stint_mats(hl, al, pairs, matches):
        rows_off, cols_off, vals_off = [], [], []
        rows_def, cols_def, vals_def = [], [], []
        rows_m, cols_m, vals_m = [], [], []
        for ri, (h, a) in enumerate(zip(hl, al)):
            # home offense: home players + home teammate pairs
            for pid in h:
                if pid in pid_to_i:
                    rows_off.append(ri); cols_off.append(pid_to_i[pid]); vals_off.append(1.0)
            hs = sorted(h)
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    k = _pair_key(hs[i], hs[j])
                    if k in pairs:
                        rows_off.append(ri); cols_off.append(p + pairs[k]); vals_off.append(1.0)
            # home defense / away offense: away off (+), home def (-)
            for pid in a:
                if pid in pid_to_i:
                    rows_def.append(ri); cols_def.append(pid_to_i[pid]); vals_def.append(1.0)
            for pid in h:
                if pid in pid_to_i:
                    rows_def.append(ri); cols_def.append(pid_to_i[pid]); vals_def.append(-1.0)
            for i in h:
                for j in a:
                    k = (i, j)
                    if k in matches:
                        rows_m.append(ri); cols_m.append(matches[k]); vals_m.append(1.0)
        X_off = sparse.csr_matrix((vals_off, (rows_off, cols_off)), shape=(n, p + len(pairs)))
        X_def = sparse.csr_matrix((vals_def, (rows_def, cols_def)), shape=(n, p))
        X_m = sparse.csr_matrix((vals_m, (rows_m, cols_m)),
                                shape=(n, len(matches))) if matches else sparse.csr_matrix((n, 0))
        return X_off, X_def, X_m

    hl = stints.home.tolist()
    al = stints.away.tolist()
    X_off, X_def, X_m = stint_mats(hl, al, pair_idx, match_idx)
    y_off = stints.Y_OFF_HOME.values.astype(np.float64)
    y_def = stints.Y_DEF_HOME.values.astype(np.float64)
    w = np.sqrt(stints.POSS.values.astype(np.float64))
    w = w / w.mean()
    return X_off, X_def, X_m, y_off, y_def, w


def box_priors(data, players, target_season):
    """Off/def prior means from box offensiveWS/defensiveWS shares."""
    bk = pd.read_csv(pi.PLAYER_DATA).dropna(subset=["box"])
    bk["nm"] = bk.playerName.map(pi.norm_name)
    pid_nm = {}
    for s in data.seasons:
        if s > target_season:
            continue
        pl = data.PLAYERS.get(s)
        if pl is None:
            continue
        for pid, nm in zip(pl.PLAYER_ID, pl.NAME):
            pid_nm[int(pid)] = pi.norm_name(nm)
    latest = (bk[bk.season <= target_season]
              .sort_values("season")
              .groupby("nm", as_index=False)
              .tail(1)
              .set_index("nm"))
    off_mu, def_mu = [], []
    for pid in players:
        nm = pid_nm.get(pid)
        if nm is None or nm not in latest.index:
            off_mu.append(0.0)
            def_mu.append(0.0)
            continue
        row = latest.loc[nm]
        ows = max(float(row.get("offensiveWS", 0) or 0), 0.05)
        dws = max(float(row.get("defensiveWS", 0) or 0), 0.05)
        tot = ows + dws
        off_mu.append(2.0 * (ows / tot - 0.5))
        def_mu.append(2.0 * (dws / tot - 0.5))
    return np.array(off_mu), np.array(def_mu)


def fit_bayesian(train_seasons, target_season, data=None):
    data = data or pi.BookerData()
    st = load_stints(data, train_seasons)
    if len(st) > MAX_STINTS:
        st = st.sample(MAX_STINTS, random_state=SUBSAMPLE_SEED)

    st["home"] = st.HOME_LINEUP.map(parse_lineup)
    st["away"] = st.AWAY_LINEUP.map(parse_lineup)
    players = sorted({p for hl in st.home for p in hl} | {p for al in st.away for p in al})
    pair_list = top_pairs(st, "team", MAX_PAIR)
    match_list = top_pairs(st, "cross", MAX_MATCH)
    pair_idx = {k: i for i, k in enumerate(pair_list)}
    match_idx = {k: i for i, k in enumerate(match_list)}

    X_off, X_def, X_m, y_off, y_def, w = build_design(st, players, pair_idx, match_idx)
    off_prior, def_prior = box_priors(data, players, target_season)
    n_players = len(players)
    n_pairs = len(pair_list)
    n_match = len(match_list)

    # dense for pymc (subsampled stints keep this tractable)
    Xo = X_off.toarray()
    Xd = X_def.toarray()
    Xm = X_m.toarray() if n_match else np.zeros((len(st), 0))

    coords = {
        "player": np.arange(n_players),
        "pair": np.arange(n_pairs) if n_pairs else ["_none"],
        "match": np.arange(n_match) if n_match else ["_none"],
        "stint": np.arange(len(st)),
    }

    with pm.Model(coords=coords) as model:
        off_mu = pm.Data("off_mu", off_prior)
        def_mu = pm.Data("def_mu", def_prior)

        tau_off = pm.HalfNormal("tau_off", 1.0)
        tau_def = pm.HalfNormal("tau_def", 1.0)
        player_off = pm.Normal("player_off", mu=off_mu, sigma=tau_off, dims="player")
        player_def = pm.Normal("player_def", mu=def_mu, sigma=tau_def, dims="player")

        tau_pair = pm.HalfNormal("tau_pair", 0.4)
        if n_pairs:
            pair_off = pm.Normal("pair_off", 0, tau_pair, dims="pair")
            syn_off = pm.math.dot(Xo[:, n_players:], pair_off)
        else:
            syn_off = 0.0

        tau_match = pm.HalfNormal("tau_match", 0.35)
        if n_match:
            match_eff = pm.Normal("match_eff", 0, tau_match, dims="match")
            match_term = pm.math.dot(Xm, match_eff)
        else:
            match_term = 0.0

        mu_off = pm.math.dot(Xo[:, :n_players], player_off) + syn_off
        mu_def = pm.math.dot(Xd, player_def) + match_term

        sigma_off = pm.HalfNormal("sigma_off", 8.0)
        sigma_def = pm.HalfNormal("sigma_def", 8.0)

        pm.Normal("y_off", mu=mu_off, sigma=sigma_off, observed=y_off,
                  dims="stint")
        pm.Normal("y_def", mu=mu_def, sigma=sigma_def, observed=y_def,
                  dims="stint")

        print(f"PyMC: {len(st)} stints, {n_players} players, {n_pairs} pair, {n_match} matchup terms")
        approx = pm.fit(n=ADVI_STEPS, method="advi", progressbar=True)
        idata = approx.sample(POST_SAMPLES, random_seed=42)

    post_off = idata.posterior["player_off"].stack(sample=("chain", "draw")).mean("sample").values
    post_def = idata.posterior["player_def"].stack(sample=("chain", "draw")).mean("sample").values
    sd_off = idata.posterior["player_off"].stack(sample=("chain", "draw")).std("sample").values
    sd_def = idata.posterior["player_def"].stack(sample=("chain", "draw")).std("sample").values

    pid_nm = {}
    for s in data.seasons:
        pl = data.PLAYERS.get(s)
        if pl is None:
            continue
        for pid, nm in zip(pl.PLAYER_ID, pl.NAME):
            pid_nm[int(pid)] = nm

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "PLAYER_ID": players,
        "NAME": [pid_nm.get(p, "") for p in players],
        "impact_off": post_off,
        "impact_def": post_def,
        "sd_off": sd_off,
        "sd_def": sd_def,
        "impact_total": post_off + post_def,
    })
    meta = {
        "target_season": target_season,
        "train_seasons": train_seasons,
        "n_stints": len(st),
        "n_players": n_players,
        "n_pairs": n_pairs,
        "n_match": n_match,
    }
    df.to_csv(OUT / f"player_posterior_{target_season}.csv", index=False)
    (OUT / f"meta_{target_season}.json").write_text(json.dumps(meta, indent=2))
    az.to_netcdf(idata, OUT / f"trace_{target_season}.nc")
    print(f"wrote {OUT}/player_posterior_{target_season}.csv")
    return df, idata


def player_impact_dict(target_season):
    path = OUT / f"player_posterior_{target_season}.csv"
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path)
    off = dict(zip(df.PLAYER_ID.astype(int), df.impact_off))
    def_ = dict(zip(df.PLAYER_ID.astype(int), df.impact_def))
    tot = dict(zip(df.PLAYER_ID.astype(int), df.impact_total))
    return off, def_, tot


if __name__ == "__main__":
    data = pi.BookerData()
    for target in range(2020, 2028):
        train = [s for s in range(target - 3, target) if s in data.STINTS]
        if len(train) < 2:
            continue
        print(f"\n=== Bayesian fit target {target} train={train} ===")
        fit_bayesian(train, target, data)
