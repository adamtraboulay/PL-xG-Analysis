from pathlib import Path

import pandas as pd
import soccerdata as sd 

understat = sd.understat.Understat(leagues = "ENG-Premier League", seasons = "2025/2026")
matches = understat.read_schedule().reset_index()

columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"]

data = matches.loc[matches["is_result"].eq(True), columns].copy()
data["date"] = pd.to_datetime(data["date"])

for column in ["home_goals", "away_goals", "home_xg", "away_xg"]:
    data[column] = pd.to_numeric(data[column], errors="raise")

data = data.sort_values("date").reset_index(drop=True)

assert len(data) == 380, f"Expected 380 matches, got {len(data)}"
assert not data.isna().any().any(), "Some values are missing"
assert not data.duplicated(["date", "home_team", "away_team"]).any(), "Duplicate matches found"

Path("data").mkdir(exist_ok=True)
data.to_csv("data/league-chemp.csv", index=False)

print(data.head())
print(f"\nSaved {len(data)} matches to data/league-chemp.csv")