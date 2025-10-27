import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
from dotenv import load_dotenv
from stats import stats_to_aggregate

load_dotenv()

PATH_TO_DATA_FILE = os.getenv("PATH_TO_DATA_FILE")

print("Loading data...")
df = pd.read_csv(PATH_TO_DATA_FILE)

print("Filtering for major league games...")
df = df[df["league"].isin(["LEC", "LCK", "LPL", "LCS", "WRLDS", "MSI"])]

print("Aggregating pro stats...")
numeric_cols = [col for col in stats_to_aggregate if col in df.columns and is_numeric_dtype(df[col])]
player_agg_stats = df.groupby('playername')[numeric_cols].mean()

print(player_agg_stats)