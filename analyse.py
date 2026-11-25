import pandas as pd

matches = pd.read_csv("data/league-chemp.csv")

# win = 3 points, draw = 1 point, loss = 0 points
matches["home_points"] = (3 * (matches["home_goals"] > matches["away_goals"]).astype(int) + (matches["home_goals"] == matches["away_goals"]).astype(int))
matches["away_points"] = (3 * (matches["away_goals"] > matches["home_goals"]).astype(int) + (matches["home_goals"] == matches["away_goals"]).astype(int))

print(matches[["home_team", "away_team", "home_goals", "away_goals", "home_points", "away_points"]].head())

home = pd.DataFrame({
    "Team": matches["home_team"],
    "Played": 1,
    "Goals_for": matches["home_goals"],
    "Goals_against": matches["away_goals"],
    "xG_for": matches["home_xg"],
    "xG_against": matches["away_xg"],
    "Points": matches["home_points"],
})

away = pd.DataFrame({
    "Team": matches["away_team"],
    "Played": 1,
    "Goals_for": matches["away_goals"],
    "Goals_against": matches["home_goals"],
    "xG_for": matches["away_xg"],
    "xG_against": matches["home_xg"],
    "Points": matches["away_points"],
})

table = pd.concat([home, away]).groupby("Team").sum()
table["Goal_difference"] = table["Goals_for"] - table["Goals_against"]
table["xG_diff"] = table["xG_for"] - table["xG_against"]
table["Finishing"] = table["Goals_for"] - table["xG_for"]
table["Keeping"] = table["xG_against"] - table["Goals_against"]

table = table.sort_values(["Points", "Goal_difference", "Goals_for"], ascending=False)

table = table[["Played", "Points",
               "Goals_for", "xG_for", "Finishing",
               "Goals_against", "xG_against", "Keeping",
               "Goal_difference", "xG_diff"]]
table = table.round(2)

table.to_csv("data/league_table.csv")
print(table.to_string())
