import sqlite3
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.static import teams


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

def extract_teams():
    log.info("Extracting Teams from api")
    teams_list = teams.get_teams()
    df = pd.DataFrame(teams_list)
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
    log.info("Extracting TeamStatisticsExtended.csv")
    df = pd.read_csv(
        DATA_DIR / "TeamStatisticsExtended.csv",
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

    # Keep only 2020-21 season and later
    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    season = df["gameDate"].apply(_season_from_date)
    before = len(df)
    df = df[season >= "2020-21"]
    log.info(f"  dropped {before - len(df):,} rows before the 2020-21 season")

    return df


def _season_from_date(date):
    if pd.isna(date):
        return None
    if date.month >= 10:
        start_year = date.year
    else:
        start_year = date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def transform_games(df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
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

    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Validate: do hometeamId/awayteamId actually exist in teams_df?
    valid_team_ids = set(teams_df["teamId"])
    used_team_ids = set(df["hometeamId"].dropna()) | set(df["awayteamId"].dropna())
    unmatched_team_ids = used_team_ids - valid_team_ids
    if unmatched_team_ids:
        log.warning(f"  -> {len(unmatched_team_ids)} teamId(s) in games not found in teams table: {sorted(unmatched_team_ids)}")

    before = len(df)
    df = df[df["hometeamId"].isin(valid_team_ids) & df["awayteamId"].isin(valid_team_ids)]
    log.info(f"  dropped {before - len(df):,} rows with unmatched team id(s) ({before:,} -> {len(df):,})")

    keep = [
        "gameId", "gameDate", "season", "gameType",
        "hometeamId", "awayteamId",
        "arenaName", "attendance"
    ]
    df = df[keep]

    df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    season = df["gameDate"].apply(_season_from_date)
    before = len(df)
    df = df[season >= "2020-21"]
    log.info(f"  dropped {before - len(df):,} rows before the 2020-21 season")

    return df

def transform_teams(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming teams")
    df = df.copy()

    df.rename(columns={
        "id": "teamId",
        "full_name": "teamName",
        "abbreviation": "teamAbbreviation",
        "city": "teamCity",
    }, inplace=True)

    keep = ["teamId", "teamCity", "teamName", "teamAbbreviation"]
    df = df[keep]

    return df

def transform_team_stats(df: pd.DataFrame, teams_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Transforming team stats")
    df = df.copy()

    # Drop rows with no usable game identity
    before = len(df)
    df = df.dropna(subset=["gameId"])
    df = df.drop_duplicates(subset=["gameId", "teamId"])
    log.info(f"  dropped {before - len(df):,} rows with missing id/date or dupes")

    df = df.dropna(subset=['reboundsOffensive', 'win', 'teamId'])

    # Keep only 2020-21 season and later
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    season = df["gameDateTimeEst"].apply(_season_from_date)
    before = len(df)
    df = df[season >= "2020-21"]
    log.info(f"  dropped {before - len(df):,} rows before the 2020-21 season")

    # Validate: do teamId/opponentTeamId exist in teams_df?
    valid_team_ids = set(teams_df["teamId"])
    used_team_ids = set(df["teamId"].dropna()) | set(df["opponentTeamId"].dropna())
    unmatched_team_ids = used_team_ids - valid_team_ids
    if unmatched_team_ids:
        log.warning(f"  -> {len(unmatched_team_ids)} teamId(s) not found in teams table: {sorted(unmatched_team_ids)}")

    before = len(df)
    df = df[df["teamId"].isin(valid_team_ids) & df["opponentTeamId"].isin(valid_team_ids)]
    log.info(f"  dropped {before - len(df):,} rows with unmatched team id(s) ({before:,} -> {len(df):,})")

    # Validate: does gameId exist in games_df?
    valid_game_ids = set(games_df["gameId"])
    stats_game_ids = set(df["gameId"].dropna())
    unmatched_game_ids = stats_game_ids - valid_game_ids
    if unmatched_game_ids:
        log.warning(f"  -> {len(unmatched_game_ids)} gameId(s) not found in games table")

    before = len(df)
    df = df[df["gameId"].isin(valid_game_ids)]
    log.info(f"  dropped {before - len(df):,} rows with unmatched gameId(s) ({before:,} -> {len(df):,})")

    keep = [
        "gameId", "gameDateTimeEst", "gameType", "gameLabel", "gameSubLabel",
        "seriesGameNumber", "teamId", "opponentTeamId", "home", "win",
        "teamScore", "opponentScore", "seed", "numMinutes", "assists",
        "steals", "blocks", "blocksAgainst", "fieldGoalsMade",
        "fieldGoalsAttempted", "fieldGoalsPercentage", "threePointersMade",
        "threePointersAttempted", "threePointersPercentage", "freeThrowsMade",
        "freeThrowsAttempted", "freeThrowsPercentage", "reboundsOffensive",
        "reboundsDefensive", "reboundsTotal", "reboundsTeam", "foulsPersonal",
        "personalFoulsDrawn", "turnovers", "turnoversTeam", "plusMinusPoints",
        "q1Points", "q2Points", "q3Points", "q4Points", "ot1Points",
        "ot2Points", "otAllPoints", "benchPoints", "biggestLead",
        "biggestScoringRun", "leadChanges", "pointsFastBreak",
        "pointsFromTurnovers", "pointsInThePaint", "pointsSecondChance",
        "timesTied", "timeoutsRemaining", "seasonWins", "seasonLosses",
        "estimatedOffensiveRating", "offensiveRating",
        "estimatedDefensiveRating", "defensiveRating", "estimatedNetRating",
        "netRating", "assistPercentage", "assistToTurnoverRatio",
        "assistRatio", "offensiveReboundPercentage",
        "defensiveReboundPercentage", "reboundPercentage",
        "teamTurnoverPercentage", "effectiveFieldGoalPercentage",
        "trueShootingPercentage", "estimatedPace", "pace", "pacePer40",
        "possessions", "playerImpactEstimate", "pointsOffTurnovers",
        "opponentPointsOffTurnovers", "opponentPointsSecondChance",
        "opponentPointsFastBreak", "opponentPointsInPaint",
        "percentFieldGoalAttempts2Point", "percentFieldGoalAttempts3Point",
        "percentPoints2Point", "percentPoints2PointMidRange",
        "percentPoints3Point", "percentPointsFastBreak",
        "percentPointsFreeThrow", "percentPointsOffTurnovers",
        "percentPointsInPaint", "percentAssisted2PointMade",
        "percentUnassisted2PointMade", "percentAssisted3PointMade",
        "percentUnassisted3PointMade", "percentAssistedFieldGoalsMade",
        "percentUnassistedFieldGoalsMade", "freeThrowAttemptRate",
        "opponentEffectiveFieldGoalPercentage", "opponentFreeThrowAttemptRate",
        "opponentTurnoverPercentage", "opponentOffensiveReboundPercentage",
    ]
    df = df[keep]

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
    conn.execute("""
        CREATE TABLE games (
            gameId INTEGER PRIMARY KEY,
            gameDate TEXT,
            season TEXT,
            gameType TEXT,
            hometeamId INTEGER,
            awayteamId INTEGER,
            arenaName TEXT,
            attendance INTEGER,
            FOREIGN KEY (hometeamId) REFERENCES teams (teamId),
            FOREIGN KEY (awayteamId) REFERENCES teams (teamId)
        )
    """)
    df.to_sql("games", conn, if_exists="append", index=False)
    conn.close()

def load_teams(df: pd.DataFrame):
    log.info("Loading teams into SQLite")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE teams (
            teamId INTEGER PRIMARY KEY,
            teamCity TEXT,
            teamName TEXT,
            teamAbbreviation TEXT
        )
    """)
    df.to_sql("teams", conn, if_exists="append", index=False)
    conn.close()

def load_team_stats(df: pd.DataFrame):
    log.info("Loading team stats into SQLite")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE team_stats (
            gameId INTEGER,
            gameDateTimeEst TEXT,
            gameType TEXT,
            gameLabel TEXT,
            gameSubLabel TEXT,
            seriesGameNumber INTEGER,
            teamId INTEGER,
            opponentTeamId INTEGER,
            home INTEGER,
            win INTEGER,
            teamScore INTEGER,
            opponentScore INTEGER,
            seed INTEGER,
            numMinutes REAL,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            blocksAgainst INTEGER,
            fieldGoalsMade INTEGER,
            fieldGoalsAttempted INTEGER,
            fieldGoalsPercentage REAL,
            threePointersMade INTEGER,
            threePointersAttempted INTEGER,
            threePointersPercentage REAL,
            freeThrowsMade INTEGER,
            freeThrowsAttempted INTEGER,
            freeThrowsPercentage REAL,
            reboundsOffensive INTEGER,
            reboundsDefensive INTEGER,
            reboundsTotal INTEGER,
            reboundsTeam INTEGER,
            foulsPersonal INTEGER,
            personalFoulsDrawn INTEGER,
            turnovers INTEGER,
            turnoversTeam INTEGER,
            plusMinusPoints INTEGER,
            q1Points INTEGER,
            q2Points INTEGER,
            q3Points INTEGER,
            q4Points INTEGER,
            ot1Points INTEGER,
            ot2Points INTEGER,
            otAllPoints INTEGER,
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
            estimatedOffensiveRating REAL,
            offensiveRating REAL,
            estimatedDefensiveRating REAL,
            defensiveRating REAL,
            estimatedNetRating REAL,
            netRating REAL,
            assistPercentage REAL,
            assistToTurnoverRatio REAL,
            assistRatio REAL,
            offensiveReboundPercentage REAL,
            defensiveReboundPercentage REAL,
            reboundPercentage REAL,
            teamTurnoverPercentage REAL,
            effectiveFieldGoalPercentage REAL,
            trueShootingPercentage REAL,
            estimatedPace REAL,
            pace REAL,
            pacePer40 REAL,
            possessions REAL,
            playerImpactEstimate REAL,
            pointsOffTurnovers INTEGER,
            opponentPointsOffTurnovers INTEGER,
            opponentPointsSecondChance INTEGER,
            opponentPointsFastBreak INTEGER,
            opponentPointsInPaint INTEGER,
            percentFieldGoalAttempts2Point REAL,
            percentFieldGoalAttempts3Point REAL,
            percentPoints2Point REAL,
            percentPoints2PointMidRange REAL,
            percentPoints3Point REAL,
            percentPointsFastBreak REAL,
            percentPointsFreeThrow REAL,
            percentPointsOffTurnovers REAL,
            percentPointsInPaint REAL,
            percentAssisted2PointMade REAL,
            percentUnassisted2PointMade REAL,
            percentAssisted3PointMade REAL,
            percentUnassisted3PointMade REAL,
            percentAssistedFieldGoalsMade REAL,
            percentUnassistedFieldGoalsMade REAL,
            freeThrowAttemptRate REAL,
            opponentEffectiveFieldGoalPercentage REAL,
            opponentFreeThrowAttemptRate REAL,
            opponentTurnoverPercentage REAL,
            opponentOffensiveReboundPercentage REAL,
            PRIMARY KEY (gameId, teamId),
            FOREIGN KEY (teamId) REFERENCES teams (teamId),
            FOREIGN KEY (opponentTeamId) REFERENCES teams (teamId),
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
    conn.execute("DROP TABLE IF EXISTS games")
    conn.execute("DROP TABLE IF EXISTS players")
    conn.execute("DROP TABLE IF EXISTS teams")
    conn.close()

    # Extract
    teams_df = extract_teams()
    players_df = extract_players()
    games_df = extract_games()
    player_stats_df = extract_player_stats()
    team_stats_df = extract_team_stats()

    # Transform
    teams_df = transform_teams(teams_df)
    games_df = transform_games(games_df, teams_df)
    players_df = transform_players(players_df)
    player_stats_df = transform_player_stats(player_stats_df, players_df, games_df)
    team_stats_df = transform_team_stats(team_stats_df, teams_df, games_df)

    # Load
    load_teams(teams_df)
    load_players(players_df)
    load_games(games_df)
    load_player_stats(player_stats_df)
    load_team_stats(team_stats_df)

if __name__ == "__main__":
    run_etl()