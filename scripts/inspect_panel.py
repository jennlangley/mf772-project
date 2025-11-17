from pathlib import Path
import pandas as pd

df = pd.read_parquet("data/processed/panel_2023Q1_small.parquet")
print(df.shape)
print(df.columns)
print(df.head())
