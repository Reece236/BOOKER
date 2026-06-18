"""
Convert nba_api PlayByPlayV3 output into the classic NBA-stats play-by-play
column layout that build_season_stints.build_lineup_stints expects.

PlayByPlayV2 is deprecated (returns empty JSON), so V3 is the only live source.
V3 carries one actor per row (personId / teamId / location) and encodes a
substitution as a single row whose personId is the player going OUT, with the
incoming player named only in the description ("SUB: <IN> FOR <OUT>"). We resolve
the incoming player's id from the per-team name->id map within the game.

Output columns (subset used downstream):
    GAME_ID, PERIOD, PCTIMESTRING, EVENTMSGTYPE, SCOREMARGIN, EVENTNUM,
    HOMEDESCRIPTION, PLAYER1_ID, PLAYER1_NAME, PLAYER1_TEAM_ID,
    PLAYER2_ID, PLAYER2_NAME, PLAYER2_TEAM_ID, PLAYER3_ID, PLAYER3_TEAM_ID
"""
import re

import numpy as np
import pandas as pd

SUB_EVENTMSGTYPE = 8
_CLOCK = re.compile(r"PT0*(\d+)M0*(\d+(?:\.\d+)?)S")
_SUB = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)")


def _clock_to_pctime(clock):
    m = _CLOCK.match(str(clock))
    if not m:
        return np.nan
    minutes = int(m.group(1))
    seconds = int(float(m.group(2)))
    return f"{minutes}:{seconds:02d}"


def convert(df3):
    """Return a classic-schema DataFrame from a single-game (or multi-game) V3 frame."""
    df = df3.copy()
    df["GAME_ID"] = df["gameId"].astype("int64")
    df["PERIOD"] = df["period"].astype(int)
    df["EVENTNUM"] = df["actionNumber"].astype(int)
    df["PCTIMESTRING"] = df["clock"].map(_clock_to_pctime)

    df["PLAYER1_ID"] = pd.to_numeric(df["personId"], errors="coerce").fillna(0).astype("int64")
    df["PLAYER1_NAME"] = df["playerName"]
    df["PLAYER1_TEAM_ID"] = pd.to_numeric(df["teamId"], errors="coerce")

    sh = pd.to_numeric(df["scoreHome"], errors="coerce")
    sa = pd.to_numeric(df["scoreAway"], errors="coerce")
    df["SCOREMARGIN"] = sh - sa

    is_home = df["location"].astype(str) == "h"
    df["DESCRIPTION"] = df["description"]
    df["HOMEDESCRIPTION"] = np.where(is_home, df["description"], np.nan)

    df["EVENTMSGTYPE"] = np.where(df["actionType"] == "Substitution", SUB_EVENTMSGTYPE, 0)

    # name -> id per (game, team) for resolving the incoming sub player
    valid = df[(df.PLAYER1_ID != 0) & df.PLAYER1_TEAM_ID.notna()]
    name_to_id = {}
    for g, t, nm, pid in zip(valid.GAME_ID, valid.PLAYER1_TEAM_ID,
                             valid.PLAYER1_NAME, valid.PLAYER1_ID):
        name_to_id.setdefault((g, t), {}).setdefault(str(nm).strip(), pid)

    p2_id = np.zeros(len(df), dtype="int64")
    p2_name = np.array([""] * len(df), dtype=object)
    p2_team = df["PLAYER1_TEAM_ID"].to_numpy(copy=True)

    sub_mask = df["EVENTMSGTYPE"].to_numpy() == SUB_EVENTMSGTYPE
    gids = df.GAME_ID.to_numpy()
    teams = df.PLAYER1_TEAM_ID.to_numpy()
    descs = df.description.to_numpy()
    for i in np.where(sub_mask)[0]:
        m = _SUB.search(str(descs[i]))
        if not m:
            continue
        in_name = m.group(1).strip()
        lookup = name_to_id.get((gids[i], teams[i]), {})
        in_id = lookup.get(in_name)
        if in_id is None:                       # tolerate minor formatting diffs
            for nm, pid in lookup.items():
                if nm and (nm in in_name or in_name in nm):
                    in_id = pid
                    break
        if in_id is not None:
            p2_id[i] = in_id
            p2_name[i] = in_name

    df["PLAYER2_ID"] = p2_id
    df["PLAYER2_NAME"] = pd.Series(p2_name).replace("", np.nan).to_numpy()
    df["PLAYER2_TEAM_ID"] = p2_team
    df["PLAYER3_ID"] = 0
    df["PLAYER3_NAME"] = np.nan
    df["PLAYER3_TEAM_ID"] = np.nan

    # PERSONxTYPE only needs to avoid the {6,7} (non-player) buckets that
    # nba_on_court filters out; any player bucket value works.
    df["PERSON1TYPE"] = np.where(df["PLAYER1_ID"] != 0, 4, 0)
    df["PERSON2TYPE"] = np.where(df["PLAYER2_ID"] != 0, 4, 0)
    df["PERSON3TYPE"] = 0

    cols = ["GAME_ID", "PERIOD", "PCTIMESTRING", "EVENTMSGTYPE", "SCOREMARGIN",
            "EVENTNUM", "DESCRIPTION", "HOMEDESCRIPTION", "PLAYER1_ID",
            "PLAYER1_NAME", "PLAYER1_TEAM_ID", "PLAYER2_ID", "PLAYER2_NAME",
            "PLAYER2_TEAM_ID", "PLAYER3_ID", "PLAYER3_NAME", "PLAYER3_TEAM_ID",
            "PERSON1TYPE", "PERSON2TYPE", "PERSON3TYPE"]
    return df[cols]
