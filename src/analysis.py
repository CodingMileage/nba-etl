import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy

DB_PATH = Path(__file__).resolve().parent.parent / "output" / "my_nba.db"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"

def get_conn():
    return sqlite3.connect(DB_PATH)

def scoring_trend_by_season():
    conn = get_conn()
    q = """
        SELECT season, ROUND(AVG(points), 2) AS avg_points_per_player_game, COUNT(*) AS n
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



def points_vs_minutes_regression_regular_season():
    conn = get_conn()
    query = f"""
        SELECT numMinutes, points
        FROM player_game_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season'
    """
    df = pd.read_sql(query, conn)
    conn.close()

    regression = scipy.stats.linregress(x=df["numMinutes"], y=df["points"])
    print(f"Minutes vs Points Regression: {regression.slope:.4f} * x + {regression.intercept:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["numMinutes"], df["points"], alpha=0.5)

    x_vals = np.array([df["numMinutes"].min(), df["numMinutes"].max()])
    y_vals = np.array([regression.slope * x + regression.intercept for x in x_vals])
    ax.plot(x_vals, y_vals, color="red", linewidth=2,
            label=f"y = {regression.slope:.4f}x + {regression.intercept:.4f} (r={regression.rvalue:.3f})")
    ax.legend()
    
    ax.set_title("Minutes vs Points (Regular Season)")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Points")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "minutes_vs_points_regression_regular_season.png")
    plt.close(fig)
    df.to_csv(OUT_DIR / "minutes_vs_points_regression_regular_season.csv", index=False)

    return df

def offensive_rebounds_vs_win_percentage():
    conn = get_conn()
    query = f"""
        SELECT teamId, season, SUM(reboundsOffensive) AS total_off_rebounds,
               SUM(win) AS total_wins, COUNT(*) AS total_games
        FROM team_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season'
        GROUP BY teamId, season
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["win_percentage"] = df["total_wins"] / df["total_games"]
    df["off_rebounds_per_game"] = df["total_off_rebounds"] / df["total_games"]

    regression = scipy.stats.linregress(x=df["off_rebounds_per_game"], y=df["win_percentage"])
    print(f"Offensive Rebounds vs Win % Regression: {regression.slope:.4f} * x + {regression.intercept:.4f}")

    

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["off_rebounds_per_game"], df["win_percentage"], alpha=0.5)

    x_vals = np.array([df["off_rebounds_per_game"].min(), df["off_rebounds_per_game"].max()])
    y_vals = np.array([regression.slope * x + regression.intercept for x in x_vals])
    ax.plot(x_vals, y_vals, color="red", linewidth=2,
            label=f"y = {regression.slope:.4f}x + {regression.intercept:.4f} (r={regression.rvalue:.3f})")
    ax.legend()
    
    ax.set_title("Offensive Rebounds per Game vs Win Percentage")
    ax.set_xlabel("Offensive Rebounds per Game")
    ax.set_ylabel("Win Percentage")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "offensive_rebounds_vs_win_percentage.png")
    plt.close(fig)
    df.to_csv(OUT_DIR / "offensive_rebounds_vs_win_percentage.csv", index=False)

    return df

def assists_vs_win_percentage():
    conn = get_conn()
    query = f"""
        SELECT teamId, season, SUM(assists) AS total_assists,
               SUM(win) AS total_wins, COUNT(*) AS total_games
        FROM team_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season'
        GROUP BY teamId, season
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["win_percentage"] = df["total_wins"] / df["total_games"]
    df["assists_per_game"] = df["total_assists"] / df["total_games"]

    regression = scipy.stats.linregress(x=df["assists_per_game"], y=df["win_percentage"])
    print(f"Assists vs Win % Regression: {regression.slope:.4f} * x + {regression.intercept:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["assists_per_game"], df["win_percentage"], alpha=0.5)

    x_vals = np.array([df["assists_per_game"].min(), df["assists_per_game"].max()])
    y_vals = np.array([regression.slope * x + regression.intercept for x in x_vals])
    ax.plot(x_vals, y_vals, color="red", linewidth=2,
            label=f"y = {regression.slope:.4f}x + {regression.intercept:.4f} (r={regression.rvalue:.3f})")
    ax.legend()

    ax.set_title("Assists per Game vs Win Percentage")
    ax.set_xlabel("Assists per Game")
    ax.set_ylabel("Win Percentage")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "assists_vs_win_percentage.png")
    plt.close(fig)
    df.to_csv(OUT_DIR / "assists_vs_win_percentage.csv", index=False)

    return df

def turnovers_vs_win_percentage():
    conn = get_conn()
    query = f"""
        SELECT teamId, season, SUM(turnovers) AS total_turnovers,
               SUM(win) AS total_wins, COUNT(*) AS total_games
        FROM team_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season'
        GROUP BY teamId, season
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["win_percentage"] = df["total_wins"] / df["total_games"]
    df["turnovers_per_game"] = df["total_turnovers"] / df["total_games"]

    regression = scipy.stats.linregress(x=df["turnovers_per_game"], y=df["win_percentage"])
    print(f"Turnovers vs Win % Regression: {regression.slope:.4f} * x + {regression.intercept:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["turnovers_per_game"], df["win_percentage"], alpha=0.5)

    x_vals = np.array([df["turnovers_per_game"].min(), df["turnovers_per_game"].max()])
    y_vals = np.array([regression.slope * x + regression.intercept for x in x_vals])
    ax.plot(x_vals, y_vals, color="red", linewidth=2,
            label=f"y = {regression.slope:.4f}x + {regression.intercept:.4f} (r={regression.rvalue:.3f})")
    ax.legend()

    ax.set_title("Turnovers per Game vs Win Percentage")
    ax.set_xlabel("Turnovers per Game")
    ax.set_ylabel("Win Percentage")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "turnovers_vs_win_percentage.png")
    plt.close(fig)
    df.to_csv(OUT_DIR / "turnovers_vs_win_percentage.csv", index=False)

    return df

def field_goal_percentage_vs_win_percentage():
    conn = get_conn()
    query = f"""
        SELECT teamId, season, SUM(fieldGoalsMade) AS total_fg_made,
               SUM(fieldGoalsAttempted) AS total_fg_attempted,
               SUM(win) AS total_wins, COUNT(*) AS total_games
        FROM team_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season'
        GROUP BY teamId, season
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["win_percentage"] = df["total_wins"] / df["total_games"]
    df["field_goal_percentage"] = df["total_fg_made"] / df["total_fg_attempted"]

    regression = scipy.stats.linregress(x=df["field_goal_percentage"], y=df["win_percentage"])
    print(f"Field Goal % vs Win % Regression: {regression.slope:.4f} * x + {regression.intercept:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["field_goal_percentage"], df["win_percentage"], alpha=0.5)

    x_vals = np.array([df["field_goal_percentage"].min(), df["field_goal_percentage"].max()])
    y_vals = np.array([regression.slope * x + regression.intercept for x in x_vals])
    ax.plot(x_vals, y_vals, color="red", linewidth=2,
            label=f"y = {regression.slope:.4f}x + {regression.intercept:.4f} (r={regression.rvalue:.3f})")
    ax.legend()
    
    ax.set_title("Field Goal Percentage vs Win Percentage")
    ax.set_xlabel("Field Goal Percentage")
    ax.set_ylabel("Win Percentage")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "field_goal_percentage_vs_win_percentage.png")
    plt.close(fig)
    df.to_csv(OUT_DIR / "field_goal_percentage_vs_win_percentage.csv", index=False)

    return df

def pace_vs_win_percentage():
    conn = get_conn()
    query = f"""
        SELECT teamId, season, SUM(pace) AS total_pace,
                SUM(win) AS total_wins, COUNT(*) AS total_games
        FROM team_stats s
        JOIN games g ON g.gameId = s.gameId
        WHERE g.gameType = 'Regular Season'
        GROUP BY teamId, season
    """
    df = pd.read_sql(query, conn)
    conn.close()

    df["win_percentage"] = df["total_wins"] / df["total_games"]
    df["pace"] = df["total_pace"] / df["total_games"]

    x = df["pace"].values
    y = df["win_percentage"].values
    n = len(x)

    regression = scipy.stats.linregress(x=x, y=y)
    print(f"Pace vs Win % Regression: {regression.slope:.4f} * x + {regression.intercept:.4f}")

    # --- Confidence intervals on slope/intercept ---
    dof = n - 2
    t_crit = scipy.stats.t.ppf(0.975, dof)  # 95% two-sided
    slope_ci = (regression.slope - t_crit * regression.stderr,
                regression.slope + t_crit * regression.stderr)
    intercept_ci = (regression.intercept - t_crit * regression.intercept_stderr,
                     regression.intercept + t_crit * regression.intercept_stderr)
    print(f"Slope 95% CI: ({slope_ci[0]:.4f}, {slope_ci[1]:.4f})")
    print(f"Intercept 95% CI: ({intercept_ci[0]:.4f}, {intercept_ci[1]:.4f})")

    # --- Confidence band around the fitted line ---
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)
    y_pred_at_x = regression.slope * x + regression.intercept
    residual_std_err = np.sqrt(np.sum((y - y_pred_at_x) ** 2) / dof)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = regression.slope * x_line + regression.intercept
    se_fit = residual_std_err * np.sqrt(1 / n + (x_line - x_mean) ** 2 / ss_x)
    ci_upper = y_line + t_crit * se_fit
    ci_lower = y_line - t_crit * se_fit

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, alpha=0.5)
    ax.plot(x_line, y_line, color="red", linewidth=2,
            label=f"y = {regression.slope:.4f}x + {regression.intercept:.4f} (r={regression.rvalue:.3f})")
    ax.fill_between(x_line, ci_lower, ci_upper, color="red", alpha=0.15, label="95% CI (mean)")
    ax.legend()

    ax.set_title("Pace vs Win Percentage")
    ax.set_xlabel("Pace")
    ax.set_ylabel("Win Percentage")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pace_vs_win_percentage.png")
    plt.close(fig)
    df.to_csv(OUT_DIR / "pace_vs_win_percentage.csv", index=False)

    return df
def run():
    trend = scoring_trend_by_season()
    regression_df = points_vs_minutes_regression_regular_season()
    off_rebounds_df = offensive_rebounds_vs_win_percentage()
    assists_df = assists_vs_win_percentage()
    turnovers_df = turnovers_vs_win_percentage()
    field_goal_df = field_goal_percentage_vs_win_percentage()
    pace_df = pace_vs_win_percentage()
    # playoffs_df = minutes_vs_points_regression_playoffs()
    # print(trend.tail())
    # print(regression_df)
    # print(playoffs_df.head())
    # print(off_rebounds_df.head())
    # print(assists_df.head())

if __name__ == "__main__":
    run()