import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DB_PATH = Path(__file__).resolve().parent.parent / "output" / "my_nba.db"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"

def get_conn():
    return sqlite3.connect(DB_PATH)

def scoring_trend_by_season():
    conn = get_conn()
    q = """
        SELECT season, AVG(points) AS avg_points_per_player_game, COUNT(*) AS n
        FROM player_game_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season' AND season IS NOT NULL
        GROUP BY season
        ORDER BY season
    """
    df = pd.read_sql(q, conn)
    conn.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["season"], df["avg_points_per_player_game"], marker="o", ms=3, lw=1.2)
    ax.set_title("League-Wide Avg Points per Player-Game, by Season")
    ax.set_xlabel("Season (starting year)")
    ax.set_ylabel("Avg points per player-game")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scoring_trend_by_season.png")
    plt.close(fig)
    df.to_csv(OUT_DIR / "scoring_trend_by_season.csv", index=False)
    return df



def minutes_vs_points_regression_regular_season():
    conn = get_conn()
    query = f"""
        SELECT numMinutes, points
        FROM player_game_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def minutes_vs_points_regression_playoffs():
    conn = get_conn()
    query = f"""
        SELECT numMinutes, points
        FROM player_game_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Playoffs'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def run():
    trend = scoring_trend_by_season()
    regression_df = minutes_vs_points_regression_regular_season()
    playoffs_df = minutes_vs_points_regression_playoffs()
    print(trend.tail())
    print(regression_df.head())
    print(playoffs_df.head())

if __name__ == "__main__":
    run()