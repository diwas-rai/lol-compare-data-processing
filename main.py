import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

PATH_TO_DATA_FILE = os.getenv("PATH_TO_DATA_FILE")

print("Loading data...")
df = pd.read_csv(PATH_TO_DATA_FILE)
print(df)
