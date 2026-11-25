import pandas as pd
from scipy.stats import poisson

matches = pd.read_csv("data/matches_2025-2026.csv")

# win = 3 points, draw = 1 point, loss = 0 points
matches["home_points"] = (3 * (matches["home_goals"] > matches["away_goals"]).astype(int) + (matches["home_goals"] == matches["away_goals"]).astype(int))
matches["away_points"] = (3 * (matches["away_goals"] > matches["home_goals"]).astype(int) + (matches["home_goals"] == matches["away_goals"]).astype(int))

def expected_points(home_xg, away_xg, max_goals=10):
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            scoreline_prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
            if home_goals > away_goals:
                home_win_prob += scoreline_prob
            elif home_goals == away_goals:
                draw_prob += scoreline_prob
            else:
                away_win_prob += scoreline_prob

    home_xpts = 3 * home_win_prob + draw_prob
    away_xpts = 3 * away_win_prob + draw_prob
    return home_xpts, away_xpts

matches[["home_xpts", "away_xpts"]] = matches.apply(
    lambda row: expected_points(row["home_xg"], row["away_xg"]),
    axis=1,
    result_type="expand",
)

home = pd.DataFrame({
    "Team": matches["home_team"],
    "Played": 1,
    "Goals_for": matches["home_goals"],
    "Goals_against": matches["away_goals"],
    "xG_for": matches["home_xg"],
    "xG_against": matches["away_xg"],
    "Points": matches["home_points"],
    "xPts": matches["home_xpts"],
})

away = pd.DataFrame({
    "Team": matches["away_team"],
    "Played": 1,
    "Goals_for": matches["away_goals"],
    "Goals_against": matches["home_goals"],
    "xG_for": matches["away_xg"],
    "xG_against": matches["home_xg"],
    "Points": matches["away_points"],
    "xPts": matches["away_xpts"],
})

table = pd.concat([home, away]).groupby("Team").sum()
table["Goal_difference"] = table["Goals_for"] - table["Goals_against"]
table["xG_diff"] = table["xG_for"] - table["xG_against"]
table["Finishing"] = table["Goals_for"] - table["xG_for"]
table["Keeping"] = table["xG_against"] - table["Goals_against"]
table["Points_vs_xPts"] = table["Points"] - table["xPts"]

table = table.sort_values(["Points", "Goal_difference", "Goals_for"], ascending=False)
table.insert(0, "Pos", range(1, len(table) + 1))
table = table.reset_index()

table = table[["Pos", "Team", "Played", "Points", "xPts", "Points_vs_xPts",
               "Goals_for", "xG_for", "Finishing",
               "Goals_against", "xG_against", "Keeping",
               "Goal_difference", "xG_diff"]]
table = table.round(2)

# Clear names are used in the code above; short football-style headers are only for display
display_names = {
    "Played": "P",
    "Points": "Pts",
    "Points_vs_xPts": "+/-xPts",
    "Goals_for": "GF",
    "xG_for": "xGF",
    "Goals_against": "GA",
    "xG_against": "xGA",
    "Goal_difference": "GD",
    "xG_diff": "xGD",
}
table = table.rename(columns=display_names)

table.to_csv("data/league_table.csv", index=False)
print(table.to_string(index=False))
