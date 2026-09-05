import sqlite3
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from nba_api.stats.static import players


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("etl")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCE_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "source_games.db"
DB_PATH = Path(__file__).resolve().parent.parent / "output" / "my_nba.db"

#
# EXTRACT
#

def extract_players():
    log.info("Extracting Players from api")
    players_list = players.get_players()
    df = pd.DataFrame(players_list)
    log.info(f"  -> {len(df):,} raw rows")
    return df

def extract_games() -> pd.DataFrame:
    log.info("Extracting Games from source database")
    conn = sqlite3.connect(SOURCE_DB_PATH)
    df = pd.read_sql("SELECT * FROM games", conn)
    conn.close()
    log.info(f"  -> {len(df):,} raw rows")
    return df

def extract_player_stats(nrows: int | None = None) -> pd.DataFrame:
    log.info("Extracting PlayerStatistics.csv")
    df = pd.read_csv(
        DATA_DIR / "PlayerStatistics.csv",
        nrows=nrows,
        low_memory=False,
    )
    log.info(f"  -> {len(df):,} raw rows")
    return df

def extract_team_stats(nrows: int | None = None) -> pd.DataFrame:
    log.info("Extracting TeamStatistics.csv")
    df = pd.read_csv(
        DATA_DIR / "TeamStatistics.csv",
        nrows=nrows,
        low_memory=False,
    )
    log.info(f"  -> {len(df):,} raw rows")
    return df

#
# TRANSFORM
#

def transform_players(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming players")
    df = df.copy()

    df.rename(columns={
        "id": "playerId",
        "first_name": "firstName",
        "last_name": "lastName",
        "is_active": "active",
    }, inplace=True)

    df.drop(columns=["full_name"], inplace=True)

    return df

def transform_player_stats(df: pd.DataFrame, players_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming player stats")
    df = df.copy()

    df.rename(columns={"personId": "playerId"}, inplace=True)

    # Validate: does every playerId in the stats table actually exist in players_df?
    valid_ids = set(players_df["playerId"])
    stats_ids = set(df["playerId"].dropna())

    unmatched_ids = stats_ids - valid_ids
    match_rate = 1 - (len(unmatched_ids) / len(stats_ids))

    log.info(f"  -> playerId match rate: {match_rate:.4%}")
    if unmatched_ids:
        log.warning(f"  -> {len(unmatched_ids):,} distinct playerId(s) in stats not found in players table")

    # Drop rows with no matching player in players_df (includes missing/NaN playerId too)
    rows_before = len(df)
    df = df[df["playerId"].isin(valid_ids)]
    rows_after = len(df)
    log.info(f"  -> dropped {rows_before - rows_after:,} rows with unmatched/missing playerId "
              f"({rows_before:,} -> {rows_after:,})")

    # Validate: does every gameId in the stats table actually exist in games_df?
    valid_game_ids = set(games_df["gameId"])
    stats_game_ids = set(df["gameId"].dropna())

    unmatched_game_ids = stats_game_ids - valid_game_ids
    game_match_rate = 1 - (len(unmatched_game_ids) / len(stats_game_ids))

    log.info(f"  -> gameId match rate: {game_match_rate:.4%}")
    if unmatched_game_ids:
        log.warning(f"  -> {len(unmatched_game_ids):,} distinct gameId(s) in stats not found in games table")

    # Drop rows with no matching game in games_df
    rows_before = len(df)
    df = df[df["gameId"].isin(valid_game_ids)]
    rows_after = len(df)
    log.info(f"  -> dropped {rows_before - rows_after:,} rows with unmatched/missing gameId "
              f"({rows_before:,} -> {rows_after:,})")

    keep = [
        "playerId", "gameId", "firstName", "lastName",
        "gameDate", "gameType", "win", "home",
        "numMinutes", "points", "assists", "blocks", "steals",
        "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
        "threePointersAttempted", "threePointersMade", "threePointersPercentage",
        "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
        "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
        "foulsPersonal", "turnovers", "plusMinusPoints"
    ]
    df = df[keep]

    df["numMinutes"] = pd.to_numeric(df["numMinutes"], errors="coerce")
    df = df.dropna(subset=['numMinutes'])
    df['numMinutes'] = df['numMinutes'].round(2)


    numTypes = df["numMinutes"].apply(type).value_counts()
    log.info(f"  -> numMinutes types:\n{numTypes}")

    # Keep only 2025-26 season and later
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    season = df["gameDate"].apply(_season_from_date)
    before = len(df)
    df = df[season >= "2025-26"]
    log.info(f"  dropped {before - len(df):,} rows before the 2025-26 season")

    return df


def _season_from_date(date):
    if pd.isna(date):
        return None
    if date.month >= 10:
        start_year = date.year
    else:
        start_year = date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def transform_games(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming games")
    df = df.copy()

    # Parse dates
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    df["season"] = df["gameDate"].apply(_season_from_date)

    # Drop rows with no usable game identity
    before = len(df)
    df = df.dropna(subset=["gameId", "gameDate"])
    df = df.drop_duplicates(subset=["gameId"])
    log.info(f"  dropped {before - len(df):,} rows with missing id/date or dupes")

    # Nulls: attendance/arena info is legitimately missing for older/preseason
    # games -- keep as NaN rather than imputing a fake value (imputing here
    # would quietly corrupt any attendance-based analysis).
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Normalize a couple of team-name typos/blank city fields
    df["hometeamCity"] = df["hometeamCity"].fillna("Unknown")
    df["awayteamCity"] = df["awayteamCity"].fillna("Unknown")

    keep = [
        "gameId", "gameDate", "season", "gameType",
        "hometeamId", "hometeamCity", "hometeamName",
        "awayteamId", "awayteamCity", "awayteamName",
        "arenaName", "attendance"
    ]
    df = df[keep]

    return df

def transform_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming team stats")
    df = df.copy()

    # Drop rows with no usable game identity
    before = len(df)
    df = df.dropna(subset=["gameId"])
    df = df.drop_duplicates(subset=["gameId"])
    log.info(f"  dropped {before - len(df):,} rows with missing id/date or dupes")

    df = df.dropna(subset=['reboundsOffensive', 'win', 'teamId'])

    # Keep only 2020-21 season and later
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    season = df["gameDate"].apply(_season_from_date)
    before = len(df)
    df = df[season >= "2020-21"]
    log.info(f"  dropped {before - len(df):,} rows before the 2020-21 season")

    return df



#
# LOAD
#

def load_players(df: pd.DataFrame):
    log.info("Loading players into SQLite")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    # conn.execute("DROP TABLE IF EXISTS players")
    conn.execute("""
        CREATE TABLE players (
            playerId INTEGER PRIMARY KEY,
            firstName TEXT,
            lastName TEXT,
            active INTEGER
        )
    """)
    df.to_sql("players", conn, if_exists="append", index=False)
    conn.close()

def load_player_stats(df: pd.DataFrame):
    log.info("Loading player stats into SQLite")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    # conn.execute("DROP TABLE IF EXISTS player_game_stats")
    conn.execute("""
        CREATE TABLE player_game_stats (
            playerId INTEGER,
            gameId INTEGER,
            firstName TEXT,
            lastName TEXT,
            gameDate TEXT,
            gameType TEXT,
            win INTEGER,
            home INTEGER,
            numMinutes REAL,
            points INTEGER,
            assists INTEGER,
            blocks INTEGER,
            steals INTEGER,
            fieldGoalsAttempted INTEGER,
            fieldGoalsMade INTEGER,
            fieldGoalsPercentage REAL,
            threePointersAttempted INTEGER,
            threePointersMade INTEGER,
            threePointersPercentage REAL,
            freeThrowsAttempted INTEGER,
            freeThrowsMade INTEGER,
            freeThrowsPercentage REAL,
            reboundsDefensive INTEGER,
            reboundsOffensive INTEGER,
            reboundsTotal INTEGER,
            foulsPersonal INTEGER,
            turnovers INTEGER,
            plusMinusPoints INTEGER,
            FOREIGN KEY (playerId) REFERENCES players (playerId),
            FOREIGN KEY (gameId) REFERENCES games (gameId)
        )
    """)
    df.to_sql("player_game_stats", conn, if_exists="append", index=False)
    conn.close()

def load_games(df: pd.DataFrame):
    log.info("Loading games into SQLite")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    # conn.execute("DROP TABLE IF EXISTS games")
    conn.execute("""
        CREATE TABLE games (
            gameId INTEGER PRIMARY KEY,
            gameDate TEXT,
            season TEXT,
            gameType TEXT,
            hometeamId INTEGER,
            hometeamCity TEXT,
            hometeamName TEXT,
            awayteamId INTEGER,
            awayteamCity TEXT,
            awayteamName TEXT,
            arenaName TEXT,
            attendance INTEGER
        )
    """)
    df.to_sql("games", conn, if_exists="append", index=False)
    conn.close()

def load_team_stats(df: pd.DataFrame):
    log.info("Loading team stats into SQLite")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    # conn.execute("DROP TABLE IF EXISTS team_stats")
    conn.execute("""
        CREATE TABLE team_stats (
            gameId INTEGER,
            gameDateTimeEst TEXT,
            teamCity TEXT,
            teamName TEXT,
            teamId INTEGER,
            opponentTeamCity TEXT,
            opponentTeamName TEXT,
            opponentTeamId INTEGER,
            home INTEGER,
            win INTEGER,
            teamScore INTEGER,
            opponentScore INTEGER,
            assists INTEGER,
            blocks INTEGER,
            steals INTEGER,
            fieldGoalsAttempted INTEGER,
            fieldGoalsMade INTEGER,
            fieldGoalsPercentage REAL,
            threePointersAttempted INTEGER,
            threePointersMade INTEGER,
            threePointersPercentage REAL,
            freeThrowsAttempted INTEGER,
            freeThrowsMade INTEGER,
            freeThrowsPercentage REAL,
            reboundsDefensive INTEGER,
            reboundsOffensive INTEGER,
            reboundsTotal INTEGER,
            foulsPersonal INTEGER,
            turnovers INTEGER,
            plusMinusPoints INTEGER,
            numMinutes REAL,
            q1Points INTEGER,
            q2Points INTEGER,
            q3Points INTEGER,
            q4Points INTEGER,
            benchPoints INTEGER,
            biggestLead INTEGER,
            biggestScoringRun INTEGER,
            leadChanges INTEGER,
            pointsFastBreak INTEGER,
            pointsFromTurnovers INTEGER,
            pointsInThePaint INTEGER,
            pointsSecondChance INTEGER,
            timesTied INTEGER,
            timeoutsRemaining INTEGER,
            seasonWins INTEGER,
            seasonLosses INTEGER,
            coachId INTEGER,
            gameType TEXT,
            gameLabel TEXT,
            gameSubLabel TEXT,
            seriesGameNumber INTEGER,
            seed INTEGER,
            reboundsTeam INTEGER,
            turnoversTeam INTEGER,
            ot1Points INTEGER,
            ot2Points INTEGER,
            otAllPoints INTEGER,
            gameDate TEXT,
            FOREIGN KEY (gameId) REFERENCES games (gameId)
        )
    """)
    df.to_sql("team_stats", conn, if_exists="append", index=False)
    conn.close()


def run_etl():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DROP TABLE IF EXISTS player_game_stats")
    conn.execute("DROP TABLE IF EXISTS team_stats")
    conn.execute("DROP TABLE IF EXISTS players")
    conn.execute("DROP TABLE IF EXISTS games")
    conn.close()

    # Extract
    players_df = extract_players()
    games_df = extract_games()
    player_stats_df = extract_player_stats()
    team_stats_df = extract_team_stats()

    # Transform
    games_df = transform_games(games_df)
    players_df = transform_players(players_df)
    player_stats_df = transform_player_stats(player_stats_df, players_df, games_df)
    team_stats_df = transform_team_stats(team_stats_df)

    # Load
    load_players(players_df)
    load_games(games_df)
    load_player_stats(player_stats_df)
    load_team_stats(team_stats_df)
    # print(team_stats_df.columns.tolist())

if __name__ == "__main__":
    run_etl()
    # conn = sqlite3.connect(DB_PATH)
    # result = pd.read_sql("""
    #     SELECT p.firstName, p.lastName, g.gameDate, g.season, s.points, s.assists, s.reboundsTotal
    #     FROM player_game_stats s
    #     JOIN players p ON s.playerId = p.playerId
    #     JOIN games g ON s.gameId = g.gameId
    #     ORDER BY g.gameDate DESC
    #     LIMIT 10
    # """, conn)
    # conn.close()

    # print(result.to_string(index=False))