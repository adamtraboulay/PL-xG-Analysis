import pandas as pd
from scipy.stats import poisson


def load_played_matches(file_name):
    matches = pd.read_csv(file_name, parse_dates=["date"])
    return matches[matches["played"]].reset_index(drop=True)


def calculate_ratings(matches):
    home = pd.DataFrame({
        "team": matches["home_team"],
        "xg_for": matches["home_xg"],
        "xg_against": matches["away_xg"],
    })
    away = pd.DataFrame({
        "team": matches["away_team"],
        "xg_for": matches["away_xg"],
        "xg_against": matches["home_xg"],
    })
    team_games = pd.concat([home, away])

    per_game = team_games.groupby("team").mean()
    league_xg_per_team_game = team_games["xg_for"].mean()

    ratings = pd.DataFrame({
        "attack": per_game["xg_for"] / league_xg_per_team_game,
        "defence": per_game["xg_against"] / league_xg_per_team_game,
    })

    league = {
        "home_xg_per_game": matches["home_xg"].mean(),
        "away_xg_per_game": matches["away_xg"].mean(),
        "goals_per_xg": (matches["home_goals"].sum() + matches["away_goals"].sum()) / (matches["home_xg"].sum() + matches["away_xg"].sum()),
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
        league["home_xg_per_game"]
        * ratings.loc[home_team, "attack"]
        * ratings.loc[away_team, "defence"]
        * league["goals_per_xg"]
    )
    away_expected_goals = (
        league["away_xg_per_game"]
        * ratings.loc[away_team, "attack"]
        * ratings.loc[home_team, "defence"]
        * league["goals_per_xg"]
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
    print(f"  Home xG per game: {league['home_xg_per_game']:.2f}")
    print(f"  Away xG per game: {league['away_xg_per_game']:.2f}")
    print(f"  Home advantage:   {league['home_xg_per_game'] / league['away_xg_per_game']:.2f}x")
    print(f"  Goals per xG:     {league['goals_per_xg']:.2f}")

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
