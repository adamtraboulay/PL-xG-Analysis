from pathlib import Path

import pandas as pd
import soccerdata as sd

SEASONS = ["2025/2026", "2026/2027"]
COLUMNS = ["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"]
NUMBER_COLUMNS = ["home_goals", "away_goals", "home_xg", "away_xg"]


def fetch_season(season):
    understat = sd.Understat(leagues="ENG-Premier League", seasons=season)
    schedule = understat.read_schedule().reset_index()

    fixtures = schedule[COLUMNS].copy()
    fixtures["played"] = schedule["is_result"].astype(bool)
    fixtures["date"] = pd.to_datetime(fixtures["date"])

    for column in NUMBER_COLUMNS:
        fixtures[column] = pd.to_numeric(fixtures[column], errors="raise")

    fixtures = fixtures.sort_values("date").reset_index(drop=True)

    played = fixtures[fixtures["played"]]
    assert len(fixtures) == 380, f"{season}: expected 380 fixtures, got {len(fixtures)}"
    assert not played[NUMBER_COLUMNS].isna().any().any(), f"{season}: a played match is missing data"
    assert not fixtures.duplicated(["home_team", "away_team"]).any(), f"{season}: duplicate fixtures found"

    return fixtures


Path("data").mkdir(exist_ok=True)

for season in SEASONS:
    fixtures = fetch_season(season)
    file_name = f"data/matches_{season.replace('/', '-')}.csv"
    fixtures.to_csv(file_name, index=False)

    played_count = fixtures["played"].sum()
    print(f"{season}: {played_count} played, {len(fixtures) - played_count} to play, saved to {file_name}")