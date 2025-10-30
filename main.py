import io
import os

import boto3
import joblib
import pandas as pd
import umap
from dotenv import load_dotenv
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from stats import stats_to_aggregate

load_dotenv()

PATH_TO_DATA_FILE = os.getenv("PATH_TO_DATA_FILE")
PRO_STATS_OUTPUT_FILE = os.getenv("PRO_STATS_OUTPUT_FILE")
SCALER_OUTPUT_FILE = os.getenv("SCALER_OUTPUT_FILE")
UMAP_MODEL_OUTPUT_FILE = os.getenv("UMAP_MODEL_OUTPUT_FILE")
UMAP_COORDS_OUTPUT_FILE = os.getenv("UMAP_COORDS_OUTPUT_FILE")

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client("s3")

print("Loading data...")
df = pd.read_csv(PATH_TO_DATA_FILE)

print("Filtering for major league games...")
df = df[df["league"].isin(["LEC", "LCK", "LCS", "WRLDS", "MSI"])]

print("Aggregating pro stats...")
numeric_cols = [
    col for col in stats_to_aggregate if col in df.columns and is_numeric_dtype(df[col])
]
player_agg_stats = df.groupby("playername")[numeric_cols].mean()
player_agg_stats["games_played"] = df.groupby("playername").size()
player_agg_stats = player_agg_stats[player_agg_stats["games_played"] > 30]

try:
    s3_key = os.path.basename(PRO_STATS_OUTPUT_FILE)
    print(f"Generating CSV for {len(player_agg_stats)} players...")
    csv_string = player_agg_stats.to_csv()

    print(f"Uploading {s3_key} to s3://{S3_BUCKET_NAME}...")
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=csv_string)
    print("Upload successful.")
except Exception as e:
    print(f"Error uploading {s3_key}: {e}")


print("Fitting scaler...")
features_to_scale = player_agg_stats.columns.drop(["games_played"])
scaler = StandardScaler()
pro_player_stats_scaled = scaler.fit_transform(player_agg_stats[features_to_scale])

try:
    s3_key = os.path.basename(SCALER_OUTPUT_FILE)
    print(f"Serialising scaler...")
    with io.BytesIO() as scaler_buffer:
        joblib.dump(scaler, scaler_buffer)
        scaler_buffer.seek(0)

        print(f"Uploading {s3_key} to s3://{S3_BUCKET_NAME}...")
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=scaler_buffer)
    print("Upload successful.")
except Exception as e:
    print(f"Error uploading {s3_key}: {e}")


print("Running UMAP... (This may take a moment)")
model_2d = umap.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
)
pro_player_2d = zip(
    player_agg_stats.index.to_list(), model_2d.fit_transform(pro_player_stats_scaled)
)
print("UMAP complete.")

try:
    s3_key = os.path.basename(UMAP_MODEL_OUTPUT_FILE)
    print(f"Serialising UMAP model...")
    with io.BytesIO() as umap_buffer:
        joblib.dump(model_2d, umap_buffer)
        umap_buffer.seek(0)

        print(f"Uploading {s3_key} to s3://{S3_BUCKET_NAME}...")
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=umap_buffer)
    print("Upload successful.")
except Exception as e:
    print(f"Error uploading {s3_key}: {e}")

try:
    s3_key = os.path.basename(UMAP_COORDS_OUTPUT_FILE)
    print(f"Serialising UMAP coordinates...")
    with io.BytesIO() as coords_buffer:
        joblib.dump(pro_player_2d, coords_buffer)
        coords_buffer.seek(0)

        print(f"Uploading {s3_key} to s3://{S3_BUCKET_NAME}...")
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=coords_buffer)
    print("Upload sucessful.")
except Exception as e:
    print(f"Error uploading {s3_key}: {e}")
