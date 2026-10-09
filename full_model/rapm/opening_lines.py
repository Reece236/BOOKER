"""BOOKER vs OPENING lines (2018-19 .. 2021-22: cache/odds_archive.json has open + close
spreads and closing moneylines).

1. ACCURACY  spread -> P(home win) via a fitted normal map (P = Phi(-spread/s), s fit on
             prior seasons); log-loss of open, close, BOOKER pre-game on identical games.
2. ATS + CLV walk-forward: BOOKER expected margin (pre-game team-net gap mapped to points
             by OLS on prior seasons) vs the OPENING spread. Bet the side BOOKER prefers when
             the disagreement >= a threshold chosen on prior seasons. Report cover rate and
             ROI at -110 (break-even 52.38%), and CLV = share of bets where the line then
             moved toward BOOKER's side by the close (no-skill baseline ~ < 50%).
Run: python opening_lines.py
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from forecast import player_impacts as pi

CACHE = pi.CACHE
NICK = {"Hawks": "ATL", "Celtics": "BOS", "Nets": "BRK", "Hornets": "CHO", "Bobcats": "CHO", "Bulls": "CHI",
        "Cavaliers": "CLE", "Mavericks": "DAL", "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GSW",
        "Rockets": "HOU", "Pacers": "IND", "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM",
        "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN", "Pelicans": "NOP", "Hornets_NO": "NOP",
        "Knicks": "NYK", "Thunder": "OKC", "Magic": "ORL", "76ers": "PHI", "Suns": "PHO",
        "Trail Blazers": "POR", "Blazers": "POR", "Kings": "SAC", "Spurs": "SAS", "Raptors": "TOR",
        "Jazz": "UTA", "Wizards": "WAS"}


def load():
    a = pd.DataFrame(json.load(open(CACHE / "odds_archive.json")))
    a["season_end"] = a.season + 1
    a = a[a.season_end.between(2019, 2022)].copy()
    a["date"] = pd.to_datetime(a.date.astype(int).astype(str)).dt.strftime("%Y-%m-%d")
    a["home"] = a.home_team.map(NICK); a["away"] = a.away_team.map(NICK)
    a = a.dropna(subset=["home", "away", "home_open_spread", "home_close_spread"])
    # the archive has game TOTALS keyed into the spread fields on ~8% of rows (+-210..243)
    a = a[(a.home_open_spread.abs() <= 25) & (a.home_close_spread.abs() <= 25)]
    rows = []
    for s in range(2019, 2023):
        G = pd.read_csv(CACHE / f"seq_games_{s}.csv")
        g = pd.read_csv(CACHE / f"games_{s}.csv")[["GAME_ID", "HOME_PTS", "AWAY_PTS", "HOME_WIN"]]
        rows.append(G.merge(g, on="GAME_ID").assign(season=s))
    G = pd.concat(rows)
    G["gap"] = G.net_home_pregame - G.net_away_pregame
    G["margin"] = G.HOME_PTS - G.AWAY_PTS
    M = G.merge(a[["date", "home", "away", "home_open_spread", "home_close_spread", "home_close_ml",
                   "away_close_ml"]], on=["date", "home", "away"])
    return M


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    M = load()
    print(f"matched games: {len(M)} ({M.season.value_counts().sort_index().to_dict()})")
    seasons = sorted(M.season.unique())
    out = []
    TH = [1.0, 2.0, 3.0, 4.0, 5.0]
    for s in seasons[1:]:
        tr, te = M[M.season < s].copy(), M[M.season == s].copy()
        sd = lambda col: minimize_scalar(lambda q: ll(norm.cdf(-tr[col] / q), tr.HOME_WIN), bounds=(5, 25), method="bounded").x
        te["p_open"] = norm.cdf(-te.home_open_spread / sd("home_open_spread"))
        te["p_close"] = norm.cdf(-te.home_close_spread / sd("home_close_spread"))
        b, a = np.polyfit(tr.gap, tr.margin, 1)
        te["exp_m"] = a + b * te.gap
        resid_sd = np.std(tr.margin - (a + b * tr.gap))
        te["p_bk"] = norm.cdf(te.exp_m / resid_sd)
        trm = a + b * tr.gap
        def sim(D, em, th):                       # ATS vs OPEN line
            edge = em - (-D.home_open_spread)      # model margin minus market's implied margin
            pick = edge.abs() >= th
            home = edge > 0
            cover = np.where(home, D.margin + D.home_open_spread > 0, D.margin + D.home_open_spread < 0)
            push = (D.margin + D.home_open_spread) == 0
            moved = np.where(home, D.home_close_spread < D.home_open_spread, D.home_close_spread > D.home_open_spread)
            k = pick & ~push
            return cover[k], moved[pick], int(k.sum())
        best = max(TH, key=lambda th: (np.mean(sim(tr, trm, th)[0]) if sim(tr, trm, th)[2] >= 100 else 0))
        cov, mv, n = sim(te, te.exp_m, best)
        out.append(te.assign(th=best))
        print(f"  {s}: threshold {best:.0f} pts -> {n} bets, cover {cov.mean():.3f}, CLV(line moved our way) {mv.mean():.3f}")
    E = pd.concat(out)
    y = E.HOME_WIN.values
    print(f"\n1. ACCURACY on {len(E)} games (walk-forward maps): open {ll(E.p_open, y):.4f} | close {ll(E.p_close, y):.4f} | BOOKER pre-game {ll(E.p_bk, y):.4f}")
    edge = E.exp_m - (-E.home_open_spread)
    pick = edge.abs() >= E.th
    home = edge > 0
    res = E.margin + E.home_open_spread
    cover = np.where(home, res > 0, res < 0)[pick & (res != 0)]
    moved = np.where(home, E.home_close_spread < E.home_open_spread, E.home_close_spread > E.home_open_spread)[pick]
    still = (E.home_close_spread == E.home_open_spread)[pick]
    roi = (cover * (100 / 110) - (~cover)).mean()
    rng = np.random.default_rng(0)
    bs = [rng.choice(cover, len(cover)).mean() for _ in range(4000)]
    print(f"2. ATS at the OPEN (walk-forward thresholds): {len(cover)} bets, cover {cover.mean():.3f} "
          f"[95% CI {np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}], ROI at -110 {roi:+.3f} (break-even 0.524)")
    print(f"   CLV: line moved TOWARD BOOKER's side {moved.mean():.3f}, unchanged {still.mean():.3f}, against {1 - moved.mean() - still.mean():.3f}")
    allmove = np.sign(E.home_close_spread - E.home_open_spread)
    agree = np.corrcoef(-(allmove), np.sign(edge))[0, 1]
    print(f"   all games: corr(sign of BOOKER edge vs open, sign of open->close move toward home) = {agree:+.3f}")


if __name__ == "__main__":
    main()
