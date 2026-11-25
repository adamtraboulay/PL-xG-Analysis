"""Test the model on a season that has already happened.

Ratings are built from the first half of 2025/26 only, then used to predict the
second half. Comparing those predictions with what actually happened shows
whether the model is any good before trusting it on the current season.
"""
import numpy as np
import pandas as pd

from model import calculate_ratings, load_played_matches, predict_match

FIXTURES_PER_SEASON = 380  # 20 teams, each playing the other 19 home and away: 20 x 19
GAMES_PER_TEAM_IN_FIRST_HALF = 19  # half of each team's 38-game season
RESULTS = ["home_win", "draw", "away_win"]


def match_result(home_goals, away_goals):
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


def points_from_result(result, side):
    if result == "draw":
        return 1
    return 3 if result == f"{side}_win" else 0


def split_by_team_games(matches, games_per_team):
    """Split into matches where both teams are within their first `games_per_team` games, and the rest.

    Splitting by date instead would give teams unequal halves whenever matches are postponed.
    """
    games_played = {}
    in_first_half = []

    for _, row in matches.iterrows():
        home_game_number = games_played.get(row["home_team"], 0) + 1
        away_game_number = games_played.get(row["away_team"], 0) + 1
        games_played[row["home_team"]] = home_game_number
        games_played[row["away_team"]] = away_game_number

        in_first_half.append(home_game_number <= games_per_team and away_game_number <= games_per_team)

    in_first_half = pd.Series(in_first_half, index=matches.index)
    return matches[in_first_half], matches[~in_first_half]


def predict_fixtures(fixtures, ratings, league):
    predictions = [
        predict_match(row["home_team"], row["away_team"], ratings, league)
        for _, row in fixtures.iterrows()
    ]
    return pd.DataFrame(predictions, index=fixtures.index)


def log_loss(predictions, results):
    """Average of -log(probability given to what actually happened). Lower is better.

    A model that is confident and right scores close to 0; one that is confident
    and wrong is punished heavily. Guessing 1/3 for everything scores 1.10.
    """
    prob_of_actual_result = [predictions.loc[i, result] for i, result in results.items()]
    # A probability of exactly 0 would give log(0) = -infinity, so keep it just above 0
    prob_of_actual_result = np.clip(prob_of_actual_result, 1e-15, 1)
    return -np.mean(np.log(prob_of_actual_result))


def accuracy(predictions, results):
    """How often the most likely outcome was the one that happened."""
    most_likely = predictions[RESULTS].idxmax(axis=1)
    return (most_likely == results).mean()


def team_rows(fixtures, home_values, away_values):
    """Turn per-match home/away values into one row per team per match."""
    home = pd.DataFrame({"team": fixtures["home_team"], "value": home_values})
    away = pd.DataFrame({"team": fixtures["away_team"], "value": away_values})
    return pd.concat([home, away])


def expected_points_by_team(fixtures, predictions):
    rows = team_rows(
        fixtures,
        3 * predictions["home_win"] + predictions["draw"],
        3 * predictions["away_win"] + predictions["draw"],
    )
    return rows.groupby("team")["value"].sum()


def actual_points_by_team(fixtures):
    results = fixtures.apply(lambda row: match_result(row["home_goals"], row["away_goals"]), axis=1)
    rows = team_rows(
        fixtures,
        results.apply(points_from_result, side="home"),
        results.apply(points_from_result, side="away"),
    )
    return rows.groupby("team")["value"].sum()


def games_by_team(fixtures):
    return pd.concat([fixtures["home_team"], fixtures["away_team"]]).value_counts()


matches = load_played_matches("data/matches_2025-2026.csv").sort_values("date").reset_index(drop=True)
assert len(matches) == FIXTURES_PER_SEASON, (
    f"The backtest needs a finished season ({FIXTURES_PER_SEASON} played matches), got {len(matches)}"
)

first_half, second_half = split_by_team_games(matches, GAMES_PER_TEAM_IN_FIRST_HALF)

unrated_teams = set(second_half["home_team"]) | set(second_half["away_team"])
unrated_teams -= set(first_half["home_team"]) | set(first_half["away_team"])
assert not unrated_teams, f"These teams have no first-half games to rate them on: {unrated_teams}"

second_half_results = second_half.apply(
    lambda row: match_result(row["home_goals"], row["away_goals"]), axis=1
)

# --- Model 1: ratings from xG ---
xg_ratings, xg_league = calculate_ratings(first_half, stat="xg")
xg_predictions = predict_fixtures(second_half, xg_ratings, xg_league)

# --- Model 2: identical model, but ratings from actual goals ---
goals_ratings, goals_league = calculate_ratings(first_half, stat="goals")
goals_predictions = predict_fixtures(second_half, goals_ratings, goals_league)

# --- Baseline: ignore the teams, use how often each result happened in the first half ---
first_half_results = first_half.apply(
    lambda row: match_result(row["home_goals"], row["away_goals"]), axis=1
)
# reindex keeps all three results, even one that never happened in the first half
base_rates = first_half_results.value_counts(normalize=True).reindex(RESULTS, fill_value=0)
baseline_predictions = pd.DataFrame(
    [base_rates.to_dict()] * len(second_half), index=second_half.index
)

print(f"=== Match-by-match predictions for the second half ({len(second_half)} matches) ===\n")
print(f"{'Model':<28}{'Log loss':>10}{'Accuracy':>10}")
for name, predictions in [
    ("xG ratings", xg_predictions),
    ("Goals ratings", goals_predictions),
    ("Baseline (no team info)", baseline_predictions),
]:
    print(f"{name:<28}{log_loss(predictions, second_half_results):>10.3f}{accuracy(predictions, second_half_results):>10.1%}")

print("\nDraws: actual rate vs average predicted probability")
print(f"  Actual:          {(second_half_results == 'draw').mean():.1%}")
print(f"  xG model:        {xg_predictions['draw'].mean():.1%}")
print(f"  Goals model:     {goals_predictions['draw'].mean():.1%}")

# --- Season-level test: how many points will each team get in the second half? ---
first_half_points = actual_points_by_team(first_half)
first_half_points_per_game = first_half_points / games_by_team(first_half)

team_table = pd.DataFrame({
    "first_half_pts": first_half_points,
    "actual_2nd_half": actual_points_by_team(second_half),
    "xg_model": expected_points_by_team(second_half, xg_predictions),
    "goals_model": expected_points_by_team(second_half, goals_predictions),
    # "Same again": a team keeps collecting points at its first-half rate.
    # Per game, so it stays fair if postponements leave teams with unequal halves.
    "same_again": first_half_points_per_game * games_by_team(second_half),
})

print("\n=== Second-half points per team: average error (lower is better) ===\n")
for column in ["xg_model", "goals_model", "same_again"]:
    error = (team_table[column] - team_table["actual_2nd_half"]).abs().mean()
    print(f"  {column:<12} {error:.1f} points per team")

team_table["xg_model_error"] = team_table["xg_model"] - team_table["actual_2nd_half"]
print("\nTeam by team (sorted by first-half points):")
print(team_table.sort_values("first_half_pts", ascending=False).round(1).to_string())
