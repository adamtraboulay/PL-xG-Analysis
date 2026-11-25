# PL xG Analysis

Premier League analysis and match prediction using expected goals (xG) from Understat.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas scipy soccerdata
```

## Usage

```bash
python get_data.py   # download match data for each season into data/
python analyse.py    # 2025/26 table with xG and expected points (xPts)
python model.py      # team attack/defence ratings and example predictions
python backtest.py   # test the model: rate on the first half of 2025/26, predict the second
```

## Backtest results (2025/26)

| Model | Log loss | Accuracy | Avg points error per team |
|---|---|---|---|
| xG ratings | 1.059 | 49.5% | 4.5 |
| Goals ratings | 1.112 | 41.1% | 6.0 |
| Base rates (no team info) | 1.103 | 39.5% | – |
| Same points as first half | – | – | 6.9 |

Unfortunately was unable to predict current 26/27 season due to Understat website being down until further notice.