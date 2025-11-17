from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PANEL_FULL = ROOT / "data" / "processed" / "panel_2023Q1_full.parquet"
MODEL_PATH = ROOT / "models" / "xgb_prepay_2023Q1_gpu.json"
FEAT_PATH = ROOT / "models" / "xgb_features_2023Q1.txt"

df = pd.read_parquet(PANEL_FULL)
print("Full panel shape:", df.shape)

num_features = [
    "age_m", "season_sin", "season_cos",
    "fico", "ltv", "dti", "orig_rate", "orig_term",
]

cat_features = ["state", "purpose", "channel"]
cat_features = [c for c in cat_features if c in df.columns]

# build feature matrix exactly like in training
X = df[num_features + cat_features].copy()
X = pd.get_dummies(X, columns=cat_features, dummy_na=False)

# load training feature order
with open(FEAT_PATH, "r") as f:
    feat_list = [line.strip() for line in f if line.strip()]

# align columns: drop extras, add missing as zeros
X = X.reindex(columns=feat_list, fill_value=0)
print("Aligned X shape:", X.shape)

# load model correctly as a Booster
booster = xgb.Booster()
booster.load_model(MODEL_PATH)
print("Booster loaded.")

# convert features to DMatrix
dX = xgb.DMatrix(X)

# predict probabilities
p = booster.predict(dX)
df["p_prepay_next"] = p

OUT_WITH_P = ROOT / "data" / "processed" / "panel_2023Q1_with_pred.parquet"
df.to_parquet(OUT_WITH_P, index=False)
print("Saved panel with predictions to", OUT_WITH_P)

# ---- aggregate to SMM & CPR by month & coupon bucket ----

# coupon bucket: round orig_rate to nearest 0.5%
df["coupon_bucket"] = (df["orig_rate"] * 2).round() / 2.0

# month key as string, e.g. "2023-01"
df["act_period_month"] = df["act_period_dt"].dt.to_period("M").astype(str)

# group by these two columns (now both are real columns)
grp = df.groupby(
    ["act_period_month", "coupon_bucket"],
    as_index=False
)["p_prepay_next"].mean()

grp.rename(columns={"p_prepay_next": "SMM_model"}, inplace=True)
grp["CPR_model"] = 1 - (1 - grp["SMM_model"]) ** 12

OUT_CPR = ROOT / "data" / "processed" / "cpr_by_month_coupon_2023Q1.csv"
grp.to_csv(OUT_CPR, index=False)
print("Saved CPR aggregation to", OUT_CPR)
print(grp.head())
