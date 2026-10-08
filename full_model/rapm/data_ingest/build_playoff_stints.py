"""Download playoff play-by-play (shufinskiy via nba_on_court, seasontype='po')
and reconstruct lineup stints -> cache/stints_po_{season}.csv.

Mirrors build_all_stints.process_season exactly (same builder, same POSS/Y
convention); playoff pbp is ~85 games/season. 2026 playoffs already live in
stints_2026.csv (the live feed carried them), so the default range is 2018-2025
-- the seasons with BookerFormer ratings but no playoff stints.

Run: python data_ingest/build_playoff_stints.py [seasons...]
"""
import sys
from pathlib import Path

import pandas as pd
import nba_on_court as noc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from build_season_stints import build_lineup_stints          # noqa: E402

PBP = ROOT / "pbp"
CACHE = ROOT / "cache"
PBP.mkdir(exist_ok=True)
SEC_PER_POSS = 28.8


def process(season):
    shuf = season - 1
    csv = PBP / f"nbastats_po_{shuf}.csv"
    if not csv.exists():
        noc.load_nba_data(path=str(PBP), seasons=shuf, data="nbastats",
                          seasontype="po", untar=True)
    if not csv.exists():
        # loader may name without the po infix; find any fresh file mentioning the year
        cands = sorted(PBP.glob(f"*po*{shuf}*.csv")) or sorted(PBP.glob(f"nbastats_{shuf}*.csv"))
        if not cands:
            print(f"  {season}: no playoff pbp file found, skipped")
            return
        csv = cands[0]
    df = pd.read_csv(csv, low_memory=False)
    df["GAME_ID"] = df["GAME_ID"].astype("int64")
    df = df[df.GAME_ID.astype(str).str.startswith("4")]      # playoffs only (no play-in)
    stints = build_lineup_stints(df)
    stints = stints[(stints.HOME_LINEUP.str.count(",") == 4)
                    & (stints.AWAY_LINEUP.str.count(",") == 4)].copy()
    stints["POSS"] = stints.DURATION_SECONDS / SEC_PER_POSS
    stints["Y"] = stints.PLUS_MINUS / stints.POSS.replace(0, float("nan")) * 100.0
    stints = stints.dropna(subset=["Y"])
    out = CACHE / f"stints_po_{season}.csv"
    stints.to_csv(out, index=False)
    print(f"  {season}: {len(stints)} playoff stints, {stints.GAME_ID.nunique()} games -> {out.name}")
    csv.unlink(missing_ok=True)                               # raw pbp is large; drop it


if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or list(range(2018, 2026))
    for s in seasons:
        process(s)
