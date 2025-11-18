from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CPR_PATH = ROOT / "data" / "processed" / "cpr_by_month_coupon_2023Q1.csv"
FNCL_PATH = ROOT / "data" / "FNCL_price_data.csv"  

# ---- load model CPR by month & coupon ----
cpr = pd.read_csv(CPR_PATH)
print("CPR table head:")
print(cpr.head())

# ---- load FNCL prices ----
fncl = pd.read_csv(FNCL_PATH)

# try to guess date column name
date_cols = [c for c in fncl.columns if c.lower().startswith("date")]
if not date_cols:
    raise ValueError("Could not find a date column in FNCL file.")
date_col = date_cols[0]

fncl[date_col] = pd.to_datetime(fncl[date_col], errors="coerce")
fncl["act_period_month"] = fncl[date_col].dt.to_period("M").astype(str)

# EXAMPLE: mapping from column name to numerical coupon
# adjust keys to match your actual FNCL column names
col_to_coupon = {
    "FNCL_7": 7.0,
    "FNCL_6": 6.0,
    "FNCL_5_5": 5.5,
    "FNCL_5": 5.0,
    "FNCL_4_5": 4.5,
    "FNCL_4": 4.0,
    "FNCL_3_5": 3.5,
    "FNCL_3": 3.0,
    "FNCL_2_5": 2.5,
}

# keep only those columns we know how to map
price_cols = [c for c in fncl.columns if c in col_to_coupon]

# reshape FNCL prices to long format
fncl_long = fncl.melt(
    id_vars=["act_period_month"],
    value_vars=price_cols,
    var_name="fncl_col",
    value_name="price"
)

fncl_long["coupon_bucket"] = fncl_long["fncl_col"].map(col_to_coupon)

# monthly average price per month/coupon
fncl_month = (
    fncl_long
    .groupby(["act_period_month", "coupon_bucket"], as_index=False)["price"]
    .mean()
)

print("FNCL monthly head:")
print(fncl_month.head())

# ---- merge model CPR with FNCL prices ----
merged = cpr.merge(
    fncl_month,
    on=["act_period_month", "coupon_bucket"],
    how="inner"
)

# (optional) extremely rough implied CPR proxy based on price deviation from par
# this is just a placeholder; you'd refine it conceptually in the writeup
# e.g., assume higher price => lower required CPR, etc.
# For now, just keep price and model CPR; the "valuation" story is qualitative.

OUT_PATH = ROOT / "data" / "processed" / "cpr_vs_fncl_2023Q1.csv"
merged.to_csv(OUT_PATH, index=False)
print("Saved merged CPR vs FNCL to", OUT_PATH)
print(merged.head())
