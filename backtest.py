"""Test the model on a season that has already happened.

Ratings are built from the first half of 2025/26 only, then used to predict the
second half. Comparing those predictions with what actually happened shows
whether the model is any good before trusting it on the current season.
"""
import numpy as np
import pandas as pd

from model import calculate_ratings, load_played_matches, predict_match

MATCHES_PER_HALF = 190  # 20 teams x 19 games / 2 teams per match


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
    return -np.mean(np.log(prob_of_actual_result))


def accuracy(predictions, results):
    """How often the most likely outcome was the one that happened."""
    most_likely = predictions[["home_win", "draw", "away_win"]].idxmax(axis=1)
    return (most_likely == results).mean()


def expected_points_by_team(fixtures, predictions):
    home = pd.DataFrame({
        "team": fixtures["home_team"],
        "points": 3 * predictions["home_win"] + predictions["draw"],
    })
    away = pd.DataFrame({
        "team": fixtures["away_team"],
        "points": 3 * predictions["away_win"] + predictions["draw"],
    })
    return pd.concat([home, away]).groupby("team")["points"].sum()


def actual_points_by_team(fixtures):
    results = fixtures.apply(lambda row: match_result(row["home_goals"], row["away_goals"]), axis=1)
    home = pd.DataFrame({
        "team": fixtures["home_team"],
        "points": results.apply(points_from_result, side="home"),
    })
    away = pd.DataFrame({
        "team": fixtures["away_team"],
        "points": results.apply(points_from_result, side="away"),
    })
    return pd.concat([home, away]).groupby("team")["points"].sum()


matches = load_played_matches("data/matches_2025-2026.csv").sort_values("date").reset_index(drop=True)
first_half = matches.iloc[:MATCHES_PER_HALF]
second_half = matches.iloc[MATCHES_PER_HALF:]

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
base_rates = first_half_results.value_counts(normalize=True)
baseline_predictions = pd.DataFrame(
    [base_rates.to_dict()] * len(second_half), index=second_half.index
)

print("=== Match-by-match predictions for the second half (190 matches) ===\n")
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
team_table = pd.DataFrame({
    "first_half_pts": first_half_points,
    "actual_2nd_half": actual_points_by_team(second_half),
    "xg_model": expected_points_by_team(second_half, xg_predictions),
    "goals_model": expected_points_by_team(second_half, goals_predictions),
    # "Same again": a team repeats its first-half points total
    "same_again": first_half_points,
})

print("\n=== Second-half points per team: average error (lower is better) ===\n")
for column in ["xg_model", "goals_model", "same_again"]:
    error = (team_table[column] - team_table["actual_2nd_half"]).abs().mean()
    print(f"  {column:<12} {error:.1f} points per team")

team_table["xg_model_error"] = team_table["xg_model"] - team_table["actual_2nd_half"]
print("\nTeam by team (sorted by first-half points):")
print(team_table.sort_values("first_half_pts", ascending=False).round(1).to_string())
