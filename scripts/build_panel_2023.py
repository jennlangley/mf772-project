from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARQ_MIN = ROOT / "data" / "parquet_min"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Q = "2023Q1"

# ---- load single quarter ----
df = pd.read_parquet(PARQ_MIN / Q)
print("Loaded", Q, "shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---- rename to clean pythonic names ----
df = df.rename(columns={
    "LOAN_ID": "loan_id",
    "ACT_PERIOD": "act_period",
    "CHANNEL": "channel",
    "ORIG_RATE": "orig_rate",
    "ORIG_UPB": "orig_upb",
    "CURRENT_UPB": "current_upb",
    "ORIG_TERM": "orig_term",
    "OLTV": "ltv",
    "DTI": "dti",
    "CSCORE_B": "fico",
    "PURPOSE": "purpose",
    "STATE": "state",
    "PRODUCT": "product",
    "Zero_Bal_Code": "zb_code",
    "ZB_DTE": "zb_date",
})

# ---- parse dates and numerics ----
ap = df["act_period"].astype(str)
mm = ap.str.slice(0, 2)
yy = ap.str.slice(2, 6)
df["act_period_dt"] = pd.to_datetime(yy + "-" + mm + "-01", errors="coerce")

df["zb_date_dt"] = pd.to_datetime(df["zb_date"], errors="coerce")

for col in ["orig_rate", "orig_upb", "current_upb", "orig_term", "ltv", "dti", "fico"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- filter to FRM and ~30y term ----
df = df[df["product"].astype(str).str.upper().eq("FRM")]
df = df[df["orig_term"].between(300, 370)]
print("After FRM & term filter:", df.shape)

# ---- define prepayment event in month t ----
zb = df["zb_code"].astype(str).str.zfill(2)
same_month = (
    df["zb_date_dt"].dt.to_period("M") == df["act_period_dt"].dt.to_period("M")
)
df["event_prepay"] = ((zb == "01") & same_month).astype(int)

# ---- stop panel after first prepay per loan ----
df = df.sort_values(["loan_id", "act_period_dt"])
df["cum_evt"] = df.groupby("loan_id")["event_prepay"].cumsum()
df = df[df["cum_evt"] <= 1].drop(columns="cum_evt")

# ---- build hazard target: y_next (prepay next month) ----
df["y_next"] = df.groupby("loan_id")["event_prepay"].shift(-1).fillna(0).astype(int)

# ---- feature engineering: age + seasonality ----
# first month per loan
first_m = df.groupby("loan_id")["act_period_dt"].transform("min")

# convert to "month index" = year*12 + month
cur_month_index   = df["act_period_dt"].dt.year * 12 + df["act_period_dt"].dt.month
start_month_index = first_m.dt.year * 12 + first_m.dt.month

df["age_m"] = (cur_month_index - start_month_index).clip(lower=0)


m = df["act_period_dt"].dt.month
df["season_sin"] = np.sin(2 * np.pi * m / 12.0)
df["season_cos"] = np.cos(2 * np.pi * m / 12.0)

# ---- keep final modeling columns ----
feature_cols = [
    "age_m", "season_sin", "season_cos",
    "fico", "ltv", "dti", "orig_rate", "orig_term",
    "state", "purpose", "channel",
]

keep_cols = ["loan_id", "act_period_dt", "y_next"] + feature_cols
panel = df[keep_cols].dropna(
    subset=["fico", "ltv", "dti", "orig_rate", "orig_term", "act_period_dt"],
    how="any",
)

print("Modeling panel shape:", panel.shape)

# ---- save full and small sample ----
full_path = OUT_DIR / "panel_2023Q1_full.parquet"
small_path = OUT_DIR / "panel_2023Q1_small.parquet"

panel.to_parquet(full_path, index=False)
print("Saved full panel to", full_path)

frac = 0.10 if len(panel) > 1_000_000 else 1.0
panel_small = panel.sample(frac=frac, random_state=42)
panel_small.to_parquet(small_path, index=False)
print("Saved small panel to", small_path, "rows:", len(panel_small))
