import sys
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
    # astype(bool) would turn any text (even "False") into True, so check the type first
    is_result = schedule["is_result"]
    assert pd.api.types.is_bool_dtype(is_result), f"{season}: is_result should be True/False, got {is_result.dtype}"
    fixtures["played"] = is_result.fillna(False).astype(bool)
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

failed_seasons = []

for season in SEASONS:
    # One season failing (e.g. Understat being down) shouldn't stop the others from saving
    try:
        fixtures = fetch_season(season)
    except Exception as error:
        print(f"{season}: FAILED - {type(error).__name__}: {error}")
        failed_seasons.append(season)
        continue

    file_name = f"data/matches_{season.replace('/', '-')}.csv"
    fixtures.to_csv(file_name, index=False)

    played_count = fixtures["played"].sum()
    print(f"{season}: {played_count} played, {len(fixtures) - played_count} to play, saved to {file_name}")

if failed_seasons:
    print(
        "\nIf a failure was a JSONDecodeError, Understat probably sent an error page and soccerdata cached it.\n"
        "Delete the matching file in ~/soccerdata/data/Understat/ before trying again."
    )
    sys.exit(1)