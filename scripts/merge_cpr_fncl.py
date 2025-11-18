from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CPR_PATH = ROOT / "data" / "processed" / "cpr_by_month_coupon_2023Q1.csv"
FNCL_PATH = ROOT / "data" / "FNCL_price_data.csv"   # adjust name if different

# ---- load model CPR by month & coupon ----
cpr = pd.read_csv(CPR_PATH)
print("CPR table head:")
print(cpr.head(), "\n")

# ---- load FNCL prices ----
fncl = pd.read_csv(FNCL_PATH)
print("Raw FNCL columns:")
print(fncl.columns.tolist(), "\n")

# 1) rename FNCL columns from verbose Bloomberg names to compact ones
rename_map = {
    "FNCL 7 ALL Mtge":   "FNCL_7",
    "FNCL 6 ALL Mtge":   "FNCL_6",
    "FNCL 5.5 ALL Mtge": "FNCL_5_5",
    "FNCL 5 ALL Mtge":   "FNCL_5",
    "FNCL 4.5 ALL Mtge": "FNCL_4_5",
    "FNCL 4 ALL Mtge":   "FNCL_4",
    "FNCL 3.5 ALL Mtge": "FNCL_3_5",
    "FNCL 3 ALL Mtge":   "FNCL_3",
    "FNCL 2.5 ALL Mtge": "FNCL_2_5",
}

fncl = fncl.rename(columns=rename_map)

# drop any duplicate columns (e.g. second "FNCL 2.5 ALL Mtge")
fncl = fncl.loc[:, ~fncl.columns.duplicated()]

print("FNCL columns after rename/drop-duplicates:")
print(fncl.columns.tolist(), "\n")

# 2) date handling
date_col = "Date"
if date_col not in fncl.columns:
    raise ValueError(f"Expected a 'Date' column in FNCL file, got: {fncl.columns.tolist()}")

fncl[date_col] = pd.to_datetime(fncl[date_col], errors="coerce")
fncl["act_period_month"] = fncl[date_col].dt.to_period("M").astype(str)

# 3) mapping from FNCL column name to coupon bucket
col_to_coupon = {
    "FNCL_7":   7.0,
    "FNCL_6":   6.0,
    "FNCL_5_5": 5.5,
    "FNCL_5":   5.0,
    "FNCL_4_5": 4.5,
    "FNCL_4":   4.0,
    "FNCL_3_5": 3.5,
    "FNCL_3":   3.0,
    "FNCL_2_5": 2.5,
}

price_cols = [c for c in fncl.columns if c in col_to_coupon]
print("Price columns detected:", price_cols, "\n")

if not price_cols:
    raise ValueError("No FNCL price columns detected after rename. Check rename_map vs actual column names.")

# 4) reshape FNCL prices to long format
fncl_long = fncl.melt(
    id_vars=["act_period_month"],
    value_vars=price_cols,
    var_name="fncl_col",
    value_name="price",
)

fncl_long["coupon_bucket"] = fncl_long["fncl_col"].map(col_to_coupon)
fncl_long = fncl_long.dropna(subset=["price", "coupon_bucket"])

# 5) monthly average price per month/coupon
fncl_month = (
    fncl_long
    .groupby(["act_period_month", "coupon_bucket"], as_index=False)["price"]
    .mean()
)

print("FNCL monthly head:")
print(fncl_month.head(), "\n")

# 6) restrict to the months present in CPR (2023Q1)
months_cpr = cpr["act_period_month"].unique().tolist()
fncl_month = fncl_month[fncl_month["act_period_month"].isin(months_cpr)]

print("FNCL monthly after restricting to CPR months:")
print(fncl_month.head(), "\n")

# 7) merge model CPR with FNCL prices
merged = cpr.merge(
    fncl_month,
    on=["act_period_month", "coupon_bucket"],
    how="inner",
)

OUT_PATH = ROOT / "data" / "processed" / "cpr_vs_fncl_2023Q1.csv"
merged.to_csv(OUT_PATH, index=False)
print("Saved merged CPR vs FNCL to", OUT_PATH)
print("Merged head:")
print(merged.head())
print("\nMerged shape:", merged.shape)
