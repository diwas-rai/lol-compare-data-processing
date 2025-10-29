import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from stats import stats_to_aggregate

load_dotenv()

PATH_TO_DATA_FILE = os.getenv("PATH_TO_DATA_FILE")
PRO_STATS_OUTPUT_FILE = os.getenv("PRO_STATS_OUTPUT_FILE")
SCALER_OUTPUT_FILE = os.getenv("SCALER_OUTPUT_FILE")

print("Loading data...")
df = pd.read_csv(PATH_TO_DATA_FILE)

print("Filtering for major league games...")
# TODO: figure out how to work with LPL stats
df = df[df["league"].isin(["LEC", "LCK", "LCS", "WRLDS", "MSI"])]

print("Aggregating pro stats...")
numeric_cols = [
    col for col in stats_to_aggregate if col in df.columns and is_numeric_dtype(df[col])
]
player_agg_stats = df.groupby("playername")[numeric_cols].mean()
player_agg_stats["games_played"] = df.groupby("playername").size()
player_agg_stats = player_agg_stats[player_agg_stats["games_played"] > 30]
print(f"Saving {len(player_agg_stats)} pro players to {PRO_STATS_OUTPUT_FILE}...")
player_agg_stats.to_csv(PRO_STATS_OUTPUT_FILE)

print("Fitting scaler...")
features_to_scale = player_agg_stats.columns.drop(["games_played"])
scaler = StandardScaler()
pro_player_stats_scaled = scaler.fit_transform(player_agg_stats[features_to_scale])
print(f"Saving scaler to {SCALER_OUTPUT_FILE}...")
joblib.dump(scaler, SCALER_OUTPUT_FILE)
