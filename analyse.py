from pathlib import Path

import pandas as pd

matches = pd.read_csv("data/league-chemp.csv")

# win = 3 points, draw = 1 point, loss = 0 points
matches["home_points"] = (3 * (matches["home_goals"] > matches["away_goals"]).astype(int) + (matches["home_goals"] == matches["away_goals"]).astype(int))
matches["away_points"] = (3 * (matches["away_goals"] > matches["home_goals"]).astype(int) + (matches["home_goals"] == matches["away_goals"]).astype(int))

print(matches[["home_team", "away_team", "home_goals", "away_goals", "home_points", "away_points"]].head())

home = pd.DataFrame({
    "team": matches["home_team"],
    "played": 1,
    "goals_for": matches["home_goals"],
    "goals_against": matches["away_goals"],
    "points": matches["home_points"],
})

away = pd.DataFrame({
    "team": matches["away_team"],
    "played": 1,
    "goals_for": matches["away_goals"],
    "goals_against": matches["home_goals"],
    "points": matches["away_points"],
})

table = pd.concat([home, away]).groupby("team").sum()
table["goal_difference"] = table["goals_for"] - table["goals_against"]

table = table.sort_values(["points", "goal_difference", "goals_for"], ascending=False,)

table.to_csv("data/league_table.csv")
print(table.to_string())