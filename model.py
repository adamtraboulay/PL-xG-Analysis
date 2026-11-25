import pandas as pd
from scipy.stats import poisson


def load_played_matches(file_name):
    matches = pd.read_csv(file_name, parse_dates=["date"])
    return matches[matches["played"]].reset_index(drop=True)


def calculate_ratings(matches, stat="xg"):
    """Rate teams from either "xg" (expected goals) or "goals" (actual goals)."""
    home_column = f"home_{stat}"
    away_column = f"away_{stat}"

    home = pd.DataFrame({
        "team": matches["home_team"],
        "for": matches[home_column],
        "against": matches[away_column],
    })
    away = pd.DataFrame({
        "team": matches["away_team"],
        "for": matches[away_column],
        "against": matches[home_column],
    })
    team_games = pd.concat([home, away])

    per_game = team_games.groupby("team").mean()
    league_per_team_game = team_games["for"].mean()

    ratings = pd.DataFrame({
        "attack": per_game["for"] / league_per_team_game,
        "defence": per_game["against"] / league_per_team_game,
    })

    total_goals = matches["home_goals"].sum() + matches["away_goals"].sum()
    total_stat = matches[home_column].sum() + matches[away_column].sum()

    league = {
        "home_per_game": matches[home_column].mean(),
        "away_per_game": matches[away_column].mean(),
        # Scales predictions to real goal levels (exactly 1.0 when rating on goals)
        "goals_scale": total_goals / total_stat,
    }
    return ratings, league


def match_probabilities(home_expected_goals, away_expected_goals, max_goals=10):
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            scoreline_prob = (
                poisson.pmf(home_goals, home_expected_goals)
                * poisson.pmf(away_goals, away_expected_goals)
            )
            if home_goals > away_goals:
                home_win_prob += scoreline_prob
            elif home_goals == away_goals:
                draw_prob += scoreline_prob
            else:
                away_win_prob += scoreline_prob

    return home_win_prob, draw_prob, away_win_prob


def predict_match(home_team, away_team, ratings, league):
    home_expected_goals = (
        league["home_per_game"]
        * ratings.loc[home_team, "attack"]
        * ratings.loc[away_team, "defence"]
        * league["goals_scale"]
    )
    away_expected_goals = (
        league["away_per_game"]
        * ratings.loc[away_team, "attack"]
        * ratings.loc[home_team, "defence"]
        * league["goals_scale"]
    )
    home_win_prob, draw_prob, away_win_prob = match_probabilities(home_expected_goals, away_expected_goals)

    return {
        "home_expected_goals": home_expected_goals,
        "away_expected_goals": away_expected_goals,
        "home_win": home_win_prob,
        "draw": draw_prob,
        "away_win": away_win_prob,
    }


if __name__ == "__main__":
    matches = load_played_matches("data/matches_2025-2026.csv")
    ratings, league = calculate_ratings(matches)

    print("League averages:")
    print(f"  Home xG per game: {league['home_per_game']:.2f}")
    print(f"  Away xG per game: {league['away_per_game']:.2f}")
    print(f"  Home advantage:   {league['home_per_game'] / league['away_per_game']:.2f}x")
    print(f"  Goals per xG:     {league['goals_scale']:.2f}")

    print("\nTeam ratings (attack: higher is better, defence: lower is better):")
    print(ratings.sort_values("attack", ascending=False).round(2).to_string())

    print("\nExample predictions:")
    for home_team, away_team in [("Arsenal", "Burnley"), ("Burnley", "Arsenal"), ("Chelsea", "Liverpool")]:
        prediction = predict_match(home_team, away_team, ratings, league)
        print(
            f"  {home_team} vs {away_team}: "
            f"xG {prediction['home_expected_goals']:.2f}-{prediction['away_expected_goals']:.2f} | "
            f"home {prediction['home_win']:.0%}, draw {prediction['draw']:.0%}, away {prediction['away_win']:.0%}"
        )
