"""
Reusable player-impact engine for the BOOKER forecasting models.

This factors the validated logic from waa_backtest.py into a small library that
all three forecasts (preseason totals, in-season updating, per-game odds) share:

  * box-prior-blended, DARKO-decayed, multi-year ridge RAPM,
  * roster aggregation to a predicted team net rating (with aging), and
  * a net-rating -> wins linear map.

The key extension over the backtest is `extra_stints`: build_impacts can fold in
current-season stints observed *up to a cutoff date* so impacts update as the
season unfolds. Stint dates come from cache/games_{season}.csv (GAME_ID -> DATE).
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
RAPM = HERE.parent
CACHE = RAPM / "cache"
ROOT = RAPM.parent.parent
PLAYER_DATA = ROOT / "full_model" / "nba_player_data_2015-2025.csv"
TEAM_PRED = ROOT / "full_model" / "team_predictions.csv"

# --- shared hyper-parameters (mirror the validated backtest) -----------------
N_PRIOR = 3
DECAY = 0.70
PRIOR_BASE = -1.0
PRIOR_K = 280.0
PRIOR_CLIP = (-12.0, 14.0)
AGE_PEAK = 27.0
AGE_QUAD = -0.03
ALPHA_GRID = [2000, 4000, 6000, 10000]
DEFAULT_ALPHA = 4000
SEC_PER_POSS = 28.8
LEAGUE_PPP100 = 108.0  # baseline points scored per 100 possessions

# game-level defaults (GAME_MARGIN_SD is re-calibrated empirically in game_odds)
HOME_COURT_ADV = 2.6          # points of home edge per game
GAME_MARGIN_SD = 13.3         # std of a single-game point margin

CONFERENCE = {
    "ATL": "E", "BOS": "E", "BRK": "E", "CHI": "E", "CHO": "E", "CLE": "E",
    "DET": "E", "IND": "E", "MIA": "E", "MIL": "E", "NYK": "E", "ORL": "E",
    "PHI": "E", "TOR": "E", "WAS": "E",
    "DAL": "W", "DEN": "W", "GSW": "W", "HOU": "W", "LAC": "W", "LAL": "W",
    "MEM": "W", "MIN": "W", "NOP": "W", "OKC": "W", "PHO": "W", "POR": "W",
    "SAC": "W", "SAS": "W", "UTA": "W",
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


class BookerData:
    """Loads and caches the per-season artifacts the impact engine needs."""

    def __init__(self, seasons=range(2015, 2028)):
        self.STINTS, self.PLAYERS, self.TEAMS, self.GAMES = {}, {}, {}, {}
        self.seasons = []
        for s in seasons:
            sp = CACHE / f"stints_{s}.csv"
            pp = CACHE / f"players_{s}.csv"
            tp = CACHE / f"teams_{s}.csv"
            if sp.exists():
                d = pd.read_csv(sp)
                d["home"] = d.HOME_LINEUP.map(parse_lineup)
                d["away"] = d.AWAY_LINEUP.map(parse_lineup)
                self.STINTS[s] = d
            elif not pp.exists():
                continue
            if pp.exists():
                self.PLAYERS[s] = pd.read_csv(pp)
            if tp.exists():
                self.TEAMS[s] = pd.read_csv(tp)
            gp = CACHE / f"games_{s}.csv"
            if gp.exists():
                self.GAMES[s] = pd.read_csv(gp)
            if s not in self.seasons:
                self.seasons.append(s)

        bk = pd.read_csv(PLAYER_DATA)
        bk["nm"] = bk.playerName.map(norm_name)
        bk = bk.dropna(subset=["box"])
        self.BOX = {(nm, ss): np.average(g.box, weights=g.minutesPlayed.clip(lower=1))
                    for (nm, ss), g in bk.groupby(["nm", "season"])}
        self.AGE = {(nm, ss): np.average(g.age, weights=g.minutesPlayed.clip(lower=1))
                    for (nm, ss), g in bk.groupby(["nm", "season"])}

        tp = pd.read_csv(TEAM_PRED)
        self.ACTUAL_WINS = {(r.team_abbr, int(r.season)): r.actual_wins
                            for r in tp.itertuples()}
        self.OLD_PRED = {(r.team_abbr, int(r.season)): r.predicted_wins
                         for r in tp.itertuples()}

        self.abbr_of = {}
        for s in self.seasons:
            for tid, ab in zip(self.TEAMS[s].TEAM_ID, self.TEAMS[s].ABBR):
                self.abbr_of[tid] = ab

        # fill actual wins from the schedule for seasons absent from the legacy
        # team_predictions file (e.g. the live 2025-26 season)
        for s, g in self.GAMES.items():
            reg = g[g.get("SEASON_TYPE", "Regular Season") == "Regular Season"]
            wins = {}
            for h, a, hw in zip(reg.HOME, reg.AWAY, reg.HOME_WIN):
                if pd.isna(hw):
                    continue
                win = h if int(hw) == 1 else a
                wins[win] = wins.get(win, 0) + 1
            for ab, w in wins.items():
                self.ACTUAL_WINS.setdefault((ab, s), w)

    # ---- date helpers -------------------------------------------------------
    def stint_dates(self, season):
        """Map GAME_ID -> DATE (str) for a season, if a schedule is available."""
        g = self.GAMES.get(season)
        if g is None:
            return {}
        return dict(zip(g.GAME_ID.astype("int64"), g.DATE))

    def season_stints_before(self, season, date):
        """Current-season stints played strictly before `date` (YYYY-MM-DD)."""
        d = self.STINTS.get(season)
        if d is None:
            return None
        dates = self.stint_dates(season)
        if not dates:
            return None
        gdate = d.GAME_ID.astype("int64").map(dates)
        return d[gdate < date].copy()


def _decayed_priors(data, train_seasons, target_season):
    wmin, wbpm, wsum, last_age = {}, {}, {}, {}
    for s in train_seasons:
        w = DECAY ** (target_season - 1 - s)
        pl = data.PLAYERS[s]
        for pid, nm, mn in zip(pl.PLAYER_ID, pl.NAME, pl.MINUTES):
            key = norm_name(nm)
            bpm = data.BOX.get((key, s))
            if bpm is None:
                continue
            bpm = min(max(bpm, PRIOR_CLIP[0]), PRIOR_CLIP[1])
            wbpm[pid] = wbpm.get(pid, 0.0) + w * mn * bpm
            wmin[pid] = wmin.get(pid, 0.0) + w * mn
            wsum[pid] = wsum.get(pid, 0.0) + mn
            ag = data.AGE.get((key, s))
            if ag is not None:
                last_age[pid] = (s, ag)
    prior = {}
    for pid in wmin:
        raw = wbpm[pid] / wmin[pid] if wmin[pid] > 0 else PRIOR_BASE
        m = wsum.get(pid, 0.0)
        sh = m / (m + PRIOR_K)
        prior[pid] = raw * sh + PRIOR_BASE * (1 - sh)
    return prior, last_age


def build_impacts(data, train_seasons, target_season, alpha=DEFAULT_ALPHA,
                  extra_stints=None, extra_weight=1.0, prior_override=None):
    """Box-prior-blended, decay-weighted ridge RAPM.

    `extra_stints` is an optional DataFrame of target-season stints (with `home`,
    `away`, `POSS`, `Y`) to fold in at `extra_weight` -- used for in-season /
    pre-game updating. `prior_override` is an optional {pid: prior_impact} dict that
    replaces the box-score (BPM) ridge prior -- e.g. the BookerFormer skill+defense
    prior -- so the in-season ridge adjusts off the better anchor. Returns
    (impact dict, prior dict, last_age dict).
    """
    prior, last_age = _decayed_priors(data, train_seasons, target_season)
    if prior_override:
        prior = {**prior, **prior_override}

    rows, cols, vals, ys, ws = [], [], [], [], []
    all_ids = set()
    for s in train_seasons:
        for hl, al in zip(data.STINTS[s].home, data.STINTS[s].away):
            all_ids.update(hl); all_ids.update(al)
    if extra_stints is not None and len(extra_stints):
        for hl, al in zip(extra_stints.home, extra_stints.away):
            all_ids.update(hl); all_ids.update(al)
    all_ids = sorted(all_ids)
    col = {p: i for i, p in enumerate(all_ids)}

    ri = 0
    for s in train_seasons:
        d = data.STINTS[s]
        dw = DECAY ** (target_season - 1 - s)
        for hl, al, poss, y in zip(d.home, d.away, d.POSS, d.Y):
            for p in hl:
                rows.append(ri); cols.append(col[p]); vals.append(1.0)
            for p in al:
                rows.append(ri); cols.append(col[p]); vals.append(-1.0)
            ys.append(y); ws.append(poss * dw); ri += 1
    if extra_stints is not None and len(extra_stints):
        for hl, al, poss, y in zip(extra_stints.home, extra_stints.away,
                                   extra_stints.POSS, extra_stints.Y):
            for p in hl:
                rows.append(ri); cols.append(col[p]); vals.append(1.0)
            for p in al:
                rows.append(ri); cols.append(col[p]); vals.append(-1.0)
            ys.append(y); ws.append(poss * extra_weight); ri += 1

    X = csr_matrix((vals, (rows, cols)), shape=(ri, len(all_ids)))
    y = np.array(ys); w = np.array(ws)
    b0 = np.array([prior.get(p, PRIOR_BASE) for p in all_ids])
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, y - X.dot(b0), sample_weight=w)
    impact = dict(zip(all_ids, b0 + ridge.coef_))
    return impact, prior, last_age


def build_impacts_off_def(data, train_seasons, target_season, alpha=DEFAULT_ALPHA,
                          extra_stints=None, extra_weight=1.0,
                          league_ppp100=LEAGUE_PPP100):
    """
    Methodical O/D split: fit separate ridge RAPM for offense and defense.

    Uses stint-level targets:
      - Y_OFF_HOME: home points/100 (home offense)
      - Y_DEF_HOME: away points/100 (home defense)

    We construct two symmetric observations per stint:
      - home offense row: lineup=home, y = Y_OFF_HOME - league
      - away offense row: lineup=away, y = Y_DEF_HOME - league

    And similarly for defense (positive is better defense):
      - home defense row: lineup=home, y = league - Y_DEF_HOME
      - away defense row: lineup=away, y = league - Y_OFF_HOME

    Priors come from the decayed box prior split using box offensive/defensive shares.
    Returns (impact_total, impact_off, impact_def, prior_total, last_age).
    """
    from . import enhanced_impacts as ei  # local import to avoid circular at module load

    prior_total, last_age = _decayed_priors(data, train_seasons, target_season)

    # universe of players
    all_ids = set()
    for s in train_seasons:
        d = data.STINTS[s]
        for hl, al in zip(d.home, d.away):
            all_ids.update(hl); all_ids.update(al)
    if extra_stints is not None and len(extra_stints):
        for hl, al in zip(extra_stints.home, extra_stints.away):
            all_ids.update(hl); all_ids.update(al)
    all_ids = sorted(all_ids)
    col = {p: i for i, p in enumerate(all_ids)}

    # prior split
    shares = ei.box_off_def_shares(data, target_season, all_ids)
    prior_off = {pid: prior_total.get(pid, PRIOR_BASE) * shares.get(pid, 0.5) for pid in all_ids}
    prior_def = {pid: prior_total.get(pid, PRIOR_BASE) * (1 - shares.get(pid, 0.5)) for pid in all_ids}
    b0_off = np.array([prior_off.get(p, PRIOR_BASE * 0.5) for p in all_ids])
    b0_def = np.array([prior_def.get(p, PRIOR_BASE * 0.5) for p in all_ids])

    # build sparse X and y for offense + defense (two rows per stint)
    rows, cols, vals = [], [], []
    y_off, w_off = [], []
    y_def, w_def = [], []

    def add_row(ri, lineup, weight, yoff, ydef):
        for p in lineup:
            if p in col:
                rows.append(ri); cols.append(col[p]); vals.append(1.0)
        y_off.append(yoff); w_off.append(weight)
        y_def.append(ydef); w_def.append(weight)

    ri = 0
    for s in train_seasons:
        d = data.STINTS[s]
        dw = DECAY ** (target_season - 1 - s)
        if "Y_OFF_HOME" not in d.columns or "Y_DEF_HOME" not in d.columns:
            raise ValueError(f"stints_{s}.csv missing Y_OFF_HOME/Y_DEF_HOME")
        for hl, al, poss, yo, yd in zip(d.home, d.away, d.POSS, d.Y_OFF_HOME, d.Y_DEF_HOME):
            wt = float(poss) * float(dw)
            # home offense / home defense
            add_row(ri, hl, wt,
                    float(yo) - league_ppp100,
                    league_ppp100 - float(yd))
            ri += 1
            # away offense / away defense
            add_row(ri, al, wt,
                    float(yd) - league_ppp100,
                    league_ppp100 - float(yo))
            ri += 1

    if extra_stints is not None and len(extra_stints):
        d = extra_stints
        if "Y_OFF_HOME" not in d.columns or "Y_DEF_HOME" not in d.columns:
            raise ValueError("extra_stints missing Y_OFF_HOME/Y_DEF_HOME")
        for hl, al, poss, yo, yd in zip(d.home, d.away, d.POSS, d.Y_OFF_HOME, d.Y_DEF_HOME):
            wt = float(poss) * float(extra_weight)
            add_row(ri, hl, wt,
                    float(yo) - league_ppp100,
                    league_ppp100 - float(yd))
            ri += 1
            add_row(ri, al, wt,
                    float(yd) - league_ppp100,
                    league_ppp100 - float(yo))
            ri += 1

    X = csr_matrix((vals, (rows, cols)), shape=(ri, len(all_ids)))
    y_off_arr = np.array(y_off, dtype=np.float64)
    y_def_arr = np.array(y_def, dtype=np.float64)
    w_arr = np.array(w_off, dtype=np.float64)

    ridge_off = Ridge(alpha=alpha, fit_intercept=True)
    ridge_def = Ridge(alpha=alpha, fit_intercept=True)
    ridge_off.fit(X, y_off_arr - X.dot(b0_off), sample_weight=w_arr)
    ridge_def.fit(X, y_def_arr - X.dot(b0_def), sample_weight=w_arr)

    off = dict(zip(all_ids, b0_off + ridge_off.coef_))
    def_ = dict(zip(all_ids, b0_def + ridge_def.coef_))
    total = {pid: off.get(pid, 0.0) + def_.get(pid, 0.0) for pid in all_ids}
    return total, off, def_, prior_total, last_age


def aged_value(impact, pid, last_age, target_season):
    """Player impact adjusted along the aging curve to the target season."""
    val = impact.get(pid, PRIOR_BASE)
    if last_age is not None and pid in last_age:
        s0, a0 = last_age[pid]
        a1 = a0 + (target_season - s0)
        val += AGE_QUAD * ((a1 - AGE_PEAK) ** 2 - (a0 - AGE_PEAK) ** 2)
    return val


def aggregate_net(data, impact, season, last_age=None, target_season=None,
                  minutes=None):
    """Predicted team net rating for `season` rosters using player impacts.

    `minutes` optionally overrides the per-player minute weights (dict pid->min);
    otherwise the season's observed minutes are used.
    """
    pl = data.PLAYERS[season]
    mins = minutes if minutes is not None else dict(zip(pl.PLAYER_ID, pl.MINUTES))
    tmin = {}
    for pid, tid in zip(pl.PLAYER_ID, pl.TEAM_ID):
        tmin[tid] = tmin.get(tid, 0.0) + mins.get(pid, 0.0)
    pred = {}
    for pid, tid in zip(pl.PLAYER_ID, pl.TEAM_ID):
        if tid not in tmin or tmin[tid] <= 0:
            continue
        val = aged_value(impact, pid, last_age, target_season) if target_season else \
            impact.get(pid, PRIOR_BASE)
        presence = mins.get(pid, 0.0) / (tmin[tid] / 5.0)
        pred[tid] = pred.get(tid, 0.0) + val * presence
    return pred


def fit_net_to_wins(data, train_seasons):
    """Linear actual-net -> actual-wins map learned on the training seasons."""
    tn, tw = [], []
    for s in train_seasons:
        for tid, net in zip(data.TEAMS[s].TEAM_ID, data.TEAMS[s].ACTUAL_NET):
            ab = data.abbr_of.get(tid)
            if (ab, s) in data.ACTUAL_WINS:
                tn.append(net); tw.append(data.ACTUAL_WINS[(ab, s)])
    k, c = np.polyfit(tn, tw, 1)
    return float(k), float(c)


def prior_train_seasons(data, target_season, n_prior=N_PRIOR, min_season=2015):
    return [s for s in range(target_season - n_prior, target_season)
            if s >= min_season and s in data.STINTS]


def pick_alpha(data, train_seasons):
    """Choose ridge alpha by predicting the most-recent training season's net."""
    val_season = max(train_seasons)
    best = None
    for alpha in ALPHA_GRID:
        impact, _, _ = build_impacts(data, train_seasons, val_season, alpha)
        pred = aggregate_net(data, impact, val_season)
        act = dict(zip(data.TEAMS[val_season].TEAM_ID, data.TEAMS[val_season].ACTUAL_NET))
        ids = [t for t in pred if t in act]
        if len(ids) < 5:
            continue
        r2 = np.corrcoef([act[t] for t in ids], [pred[t] for t in ids])[0, 1] ** 2
        if best is None or r2 > best[1]:
            best = (alpha, r2)
    return best[0] if best else DEFAULT_ALPHA
