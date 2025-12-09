#!/usr/bin/env python3
"""
Mortgage Prepayment Project - FIXED VERSION
Logistic regression pipeline (event-based, y_next).
Sits in project root: mf772-project/prepay_pipeline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PARQUET_MIN_DIR = DATA_DIR / "parquet_min"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

QUARTERS = ["2022Q1", "2022Q2", "2022Q3", "2022Q4",
            "2023Q1", "2023Q2", "2023Q3", "2023Q4"]

# Sampling fraction when loading very large quarterly files
SAMPLE_FRAC = 0.03  # 3% of loans per quarter

# Market mortgage rates (Freddie Mac PMMS, 30-year fixed)
MARKET_RATES = {
    "2022-01": 3.45, "2022-02": 3.76, "2022-03": 4.17,
    "2022-04": 4.98, "2022-05": 5.25, "2022-06": 5.70,
    "2022-07": 5.41, "2022-08": 5.22, "2022-09": 6.02,
    "2022-10": 6.90, "2022-11": 6.58, "2022-12": 6.27,
    "2023-01": 6.09, "2023-02": 6.32, "2023-03": 6.42,
    "2023-04": 6.35, "2023-05": 6.57, "2023-06": 6.71,
    "2023-07": 6.78, "2023-08": 7.07, "2023-09": 7.18,
    "2023-10": 7.63, "2023-11": 7.44, "2023-12": 6.95,
}

print("=" * 80)
print("MORTGAGE PREPAYMENT MODEL - LOGISTIC PIPELINE (FIXED)")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD PANEL DATA FROM PARQUET_MIN (SAMPLED)
# ============================================================================
print("\n[1] Loading data from all quarters (parquet_min, sampled)...")

all_dfs = []

for Q in QUARTERS:
    parq_path = PARQUET_MIN_DIR / f"{Q}.parquet"
    if not parq_path.exists():
        print(f"  ⚠ Skipping {Q}: {parq_path} not found")
        continue

    print(f"  → Reading {parq_path} ...")
    df_q = pd.read_parquet(parq_path)

    # Simple row-level sampling to control memory
    if SAMPLE_FRAC < 1.0:
        df_q = df_q.sample(frac=SAMPLE_FRAC, random_state=42).reset_index(drop=True)

    df_q["quarter"] = Q
    print(f"    ✓ Loaded {Q}: {len(df_q):,} rows after sampling")
    all_dfs.append(df_q)


if not all_dfs:
    raise ValueError(f"No data found in {PARQUET_MIN_DIR}")

df = pd.concat(all_dfs, ignore_index=True)
print(f"\n  Combined shape after sampling: {df.shape}")

# ============================================================================
# STEP 2: BASIC CLEANING, RENAMING, AND TYPE CONVERSION
# ============================================================================
print("\n[2] Data cleaning and type conversion...")

# Harmonise column names (works even if some columns are missing)
rename_map = {
    "LOAN_ID": "loan_id",
    "SERVICER": "servicer",
    "ORIG_RATE": "orig_rate",
    "CURR_RATE": "curr_rate",
    "ORIG_UPB": "orig_upb",
    "CURRENT_UPB": "current_upb",
    "ORIG_TERM": "orig_term",
    "REM_MONTHS": "remain_term",
    "ORIG_DATE": "orig_date",
    "FIRST_PAY": "first_pay_date",
    "MATR_DT": "maturity_date",
    "ACT_PERIOD": "act_period",
    "CHANNEL": "channel",
    "PURPOSE": "purpose",
    "STATE": "state",
    "PROP": "property_type",
    "OCC_STAT": "occupancy",
    "NUM_BO": "num_borrowers",
    "DTI": "dti",
    "CSCORE_B": "fico",
    "OLTV": "ltv",
    "OCLTV": "cltv",
    "Zero_Bal_Code": "zb_code",
    "ZB_DTE": "zb_date",
    "PRODUCT": "product",
}

df = df.rename(columns=rename_map)

# Filter to 30-year fixed-rate mortgages if product column is available
if "product" in df.columns:
    df = df[df["product"].astype(str).str.upper() == "FRM"]
print(f"  After FRM filter: {len(df):,} rows")

# ============================================================================
# FIX: ROBUST DATE PARSING
# ============================================================================

# ACT_PERIOD: format is MMYYYY (e.g., "012022" = January 2022)
if "act_period" in df.columns:
    ap = df["act_period"].astype(str).str.zfill(6)  # Ensure 6 digits
    mm = ap.str[:2]
    yyyy = ap.str[2:6]
    df["act_period_dt"] = pd.to_datetime(yyyy + "-" + mm + "-01", errors="coerce")
else:
    raise ValueError("act_period column not found!")

# ZB_DTE: same format MMYYYY
if "zb_date" in df.columns:
    zb = df["zb_date"].astype(str).str.zfill(6)
    zb_mm = zb.str[:2]
    zb_yyyy = zb.str[2:6]
    df["zb_date_dt"] = pd.to_datetime(zb_yyyy + "-" + zb_mm + "-01", errors="coerce")

# Other dates (if present)
for col in ["orig_date", "first_pay_date", "maturity_date"]:
    if col in df.columns:
        df[col + "_dt"] = pd.to_datetime(df[col].astype(str), format="%Y%m", errors="coerce")

# Year-month string used for market rates and aggregation
df["year_month"] = df["act_period_dt"].dt.to_period("M").astype(str)

print(f"  ✓ Parsed dates successfully")

# ============================================================================
# STEP 3: DEFINE PREPAYMENT EVENT y_next (ONE-MONTH-AHEAD)
# ============================================================================
print("\n[3] Defining prepayment events...")

# Convert numeric columns
for col in ["orig_rate", "curr_rate", "orig_upb", "current_upb", "orig_term", 
            "remain_term", "ltv", "cltv", "dti", "fico"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Sort by loan and time
df = df.sort_values(["loan_id", "act_period_dt"])

# METHOD 1: Use Zero Balance Code (simpler and more reliable)
# Zero balance code "01" = prepayment
zb_code = df["zb_code"].astype(str).str.zfill(2)
if "zb_date_dt" in df.columns:
    same_month = (df["zb_date_dt"].dt.to_period("M") == df["act_period_dt"].dt.to_period("M"))
    df["event_prepay"] = ((zb_code == "01") & same_month).astype(int)
else:
    # Fallback: just use zero balance code
    df["event_prepay"] = (zb_code == "01").astype(int)

# Create next-month target
df["y_next"] = df.groupby("loan_id")["event_prepay"].shift(-1).fillna(0).astype(int)

# Stop tracking loans after first prepayment
df["cum_prepay"] = df.groupby("loan_id")["event_prepay"].cumsum()
df = df[df["cum_prepay"] <= 1].drop(columns="cum_prepay")

print(f"  Prepayment rate (y_next): {df['y_next'].mean():.4%}")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
print("\n[4] Feature engineering...")

# --- Loan age (seasoning) ---
first_period = df.groupby("loan_id")["act_period_dt"].transform("min")
cur_idx = df["act_period_dt"].dt.year * 12 + df["act_period_dt"].dt.month
start_idx = first_period.dt.year * 12 + first_period.dt.month
df["age_m"] = (cur_idx - start_idx).clip(lower=0)
df["seasoning_ramp"] = (df["age_m"] / 30).clip(lower=0, upper=1)

# --- Seasonality ---
month = df["act_period_dt"].dt.month
df["season_sin"] = np.sin(2 * np.pi * month / 12)
df["season_cos"] = np.cos(2 * np.pi * month / 12)
df["is_summer"] = month.isin([6, 7, 8]).astype(int)

# --- Market refi incentive and rate ratio ---
df["market_rate"] = df["year_month"].map(MARKET_RATES)
df["market_rate"] = df.groupby(df["act_period_dt"].dt.year)["market_rate"].ffill().bfill()

df["refi_incentive"] = (df["orig_rate"] - df["market_rate"]).clip(-5, 5)
df["refi_incentive_sigmoid"] = 1 / (1 + np.exp(-2 * (df["refi_incentive"] - 0.5)))
df["rate_ratio"] = (df["orig_rate"] / df["market_rate"]).clip(0.5, 2.0)

# --- Burnout: number of prior months with positive incentive ---
df["in_money"] = (df["refi_incentive"] > 0.5).astype(int)
df["burnout_months"] = df.groupby("loan_id")["in_money"].cumsum().clip(upper=120)
df["burnout_factor"] = 1 / (1 + df["burnout_months"] / 12)

# --- LTV measures ---
# Assume property value = orig_upb / (ltv/100)
prop_val = df["orig_upb"] / (df["ltv"] / 100)
df["current_ltv"] = ((df["current_upb"] / prop_val) * 100).clip(upper=200)
df["log_orig_upb"] = np.log1p(df["orig_upb"])

# --- Interactions ---
df["fico_x_incentive"] = df["fico"] * df["refi_incentive"]
df["age_x_incentive"] = df["age_m"] * df["refi_incentive"]
df["upb_x_incentive"] = df["log_orig_upb"] * df["refi_incentive"]

# --- Binned features for non-linearity ---
df["ltv_bin"] = pd.cut(
    df["ltv"],
    bins=[0, 60, 70, 80, 90, 100, 200],
    labels=[1, 2, 3, 4, 5, 6]
).astype(float)

df["rate_ratio_bin"] = pd.cut(
    df["rate_ratio"],
    bins=[0.0, 0.8, 0.9, 1.0, 1.1, 2.0],
    labels=[1, 2, 3, 4, 5]
).astype(float)

print(f"  ✓ Created {df.shape[1]} total columns")

# ============================================================================
# STEP 5: FEATURE SELECTION FOR MODELING
# ============================================================================
print("\n[5] Selecting final feature set...")

num_features = [
    # Core prepayment drivers
    "refi_incentive",
    "refi_incentive_sigmoid",
    "rate_ratio",
    "burnout_months",
    "burnout_factor",
    
    # Loan characteristics
    "age_m",
    "seasoning_ramp",
    "ltv",
    "current_ltv",
    "log_orig_upb",
    "fico",
    "dti",
    
    # Seasonality
    "season_sin",
    "season_cos",
    "is_summer",
    
    # Interactions
    "fico_x_incentive",
    "age_x_incentive",
    "upb_x_incentive",
    
    # Bins
    "ltv_bin",
    "rate_ratio_bin",
]

cat_features = ["state", "purpose", "channel"]
cat_features = [c for c in cat_features if c in df.columns]

keep_cols = ["loan_id", "act_period_dt", "year_month", "orig_rate", "y_next"] + num_features + cat_features
df_model = df[keep_cols].copy()

# Drop rows with missing critical features
df_model = df_model.dropna(subset=num_features, how="any")
print(f"  Final modeling dataset: {len(df_model):,} rows")
print(f"  Prepayment rate: {df_model['y_next'].mean():.4%}")

panel_path = PROCESSED_DIR / "panel_consolidated_2022_2023.parquet"
df_model.to_parquet(panel_path, index=False)
print(f"  ✓ Saved to {panel_path}")

# ============================================================================
# STEP 6: TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================
print("\n[6] Training Logistic Regression prepayment model...")

# One-hot encode categoricals
X = df_model[num_features + cat_features].copy()
X = pd.get_dummies(X, columns=cat_features, drop_first=True)
y = df_model["y_next"].values

# Time-based train/test split (chronological)
# Reset index to avoid mismatch issues
X = X.reset_index(drop=True)
df_model = df_model.reset_index(drop=True)

cutoff_date = df_model["act_period_dt"].quantile(0.8)
train_mask = df_model["act_period_dt"] <= cutoff_date
test_mask = ~train_mask

X_train = X[train_mask].values
X_test = X[test_mask].values
y_train = df_model.loc[train_mask, "y_next"].values
y_test = df_model.loc[test_mask, "y_next"].values

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
logit = LogisticRegression(
    max_iter=500,
    solver="lbfgs",
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

print("  Training (this may take a moment)...")
logit.fit(X_train_scaled, y_train)

# Evaluate
y_pred_proba = logit.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n  ✓ Loan-level AUC: {auc:.4f}")
print(f"  (0.63-0.67 is typical and GOOD for prepayment!)")
print(f"  Train set: {len(y_train):,}, Test set: {len(y_test):,}")

# Feature importance
feature_names = X.columns.tolist()
coefs = logit.coef_.ravel()
fi = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
fi = fi.sort_values("coefficient", key=np.abs, ascending=False)

fi_path = PROCESSED_DIR / "feature_importance_logit.csv"
fi.to_csv(fi_path, index=False)
print(f"\n  Top 15 Most Important Features:")
print(fi.head(15).to_string(index=False))
print(f"\n  ✓ Saved to {fi_path}")

# Predict on full dataset (in chunks to avoid memory error)
print("\n  Scoring full dataset (in chunks)...")
chunk_size = 100_000
n_chunks = int(np.ceil(len(X) / chunk_size))
predictions = []

for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(X))
    X_chunk = X.iloc[start_idx:end_idx].values
    X_chunk_scaled = scaler.transform(X_chunk)
    preds = logit.predict_proba(X_chunk_scaled)[:, 1]
    predictions.append(preds)
    if (i + 1) % 10 == 0:
        print(f"    Processed {end_idx:,} / {len(X):,} rows...")

df_model["smm_pred"] = np.concatenate(predictions)

# ============================================================================
# STEP 7: POOL-LEVEL ACCURACY CHECK
# ============================================================================
print("\n[7] Generating pool-level CPR (what matters for trading)...")

# Coupon bucket: half-point buckets
df_model["coupon_bucket"] = (np.round(df_model["orig_rate"] * 2) / 2).astype(float)

monthly_cpr = df_model.groupby(["year_month", "coupon_bucket"]).agg(
    smm_model=("smm_pred", "mean"),
    smm_actual=("y_next", "mean"),
    n_loans=("loan_id", "count"),
).reset_index()

# Convert SMM to CPR
monthly_cpr["cpr_model"] = (1 - (1 - monthly_cpr["smm_model"]) ** 12) * 100
monthly_cpr["cpr_actual"] = (1 - (1 - monthly_cpr["smm_actual"]) ** 12) * 100

# Pool-level correlation
pool_corr = monthly_cpr[["cpr_model", "cpr_actual"]].corr().iloc[0, 1]
print(f"\n  *** POOL-LEVEL CORRELATION: {pool_corr:.3f} ***")
print("  (This is what matters for relative value trading!)")

cpr_path = PROCESSED_DIR / "cpr_by_month_coupon_2022_2023.csv"
monthly_cpr.to_csv(cpr_path, index=False)
print(f"\n  ✓ Generated CPR for {len(monthly_cpr):,} month-coupon pools")
print("\n  Sample CPR Output:")
print(monthly_cpr.head(10).to_string(index=False))
print(f"\n  ✓ Saved to {cpr_path}")

# ============================================================================
# STEP 8: MERGE WITH FNCL PRICE DATA
# ============================================================================
print("\n[8] Merging with FNCL price data...")

fncl_path = DATA_DIR / "FNCL_price_data.csv"
if not fncl_path.exists():
    print(f"  ⚠ FNCL price data not found. Skipping trading simulation.")
    print(f"  Place CSV at: {fncl_path}")
else:
    fncl = pd.read_csv(fncl_path)
    
    # Rename FNCL columns
    rename_map = {
        "FNCL 7 ALL Mtge": "FNCL_7", "FNCL 6 ALL Mtge": "FNCL_6",
        "FNCL 5.5 ALL Mtge": "FNCL_5_5", "FNCL 5 ALL Mtge": "FNCL_5",
        "FNCL 4.5 ALL Mtge": "FNCL_4_5", "FNCL 4 ALL Mtge": "FNCL_4",
        "FNCL 3.5 ALL Mtge": "FNCL_3_5", "FNCL 3 ALL Mtge": "FNCL_3",
        "FNCL 2.5 ALL Mtge": "FNCL_2_5",
    }
    
    fncl = fncl.rename(columns=rename_map)
    fncl = fncl.loc[:, ~fncl.columns.duplicated()]
    fncl["Date"] = pd.to_datetime(fncl["Date"], errors="coerce")
    fncl["year_month"] = fncl["Date"].dt.to_period("M").astype(str)
    
    col_to_coupon = {
        "FNCL_7": 7.0, "FNCL_6": 6.0, "FNCL_5_5": 5.5, "FNCL_5": 5.0,
        "FNCL_4_5": 4.5, "FNCL_4": 4.0, "FNCL_3_5": 3.5, "FNCL_3": 3.0,
        "FNCL_2_5": 2.5,
    }
    
    price_cols = [c for c in fncl.columns if c in col_to_coupon]
    
    fncl_long = fncl.melt(
        id_vars=["year_month"],
        value_vars=price_cols,
        var_name="fncl_col",
        value_name="price"
    )
    
    fncl_long["coupon_bucket"] = fncl_long["fncl_col"].map(col_to_coupon)
    fncl_long = fncl_long.dropna(subset=["price", "coupon_bucket"])
    
    fncl_monthly = fncl_long.groupby(
        ["year_month", "coupon_bucket"], as_index=False
    )["price"].mean()
    
    merged = monthly_cpr.merge(
        fncl_monthly,
        on=["year_month", "coupon_bucket"],
        how="inner"
    )
    
    merged_path = PROCESSED_DIR / "cpr_vs_fncl_2022_2023.csv"
    merged.to_csv(merged_path, index=False)
    
    print(f"  ✓ Merged CPR with FNCL prices")
    print(f"  ✓ Saved to {merged_path}")
    
    # ========================================================================
    # STEP 9: TRADING SIMULATION
    # ========================================================================
    print("\n[9] Running trading simulation...")
    
    df_trade = merged.sort_values(["coupon_bucket", "year_month"])
    
    # Market-implied CPR (stylized: inverse of price z-score)
    df_trade["price_z"] = df_trade.groupby("coupon_bucket")["price"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    df_trade["cpr_implied_market"] = df_trade["cpr_model"].mean() - (df_trade["price_z"] * 5)
    
    # Signal: model CPR < market CPR = underpriced
    df_trade["signal"] = df_trade["cpr_model"] - df_trade["cpr_implied_market"]
    df_trade["position"] = np.where(df_trade["signal"] < -2, 1, 0)  # Long
    df_trade["position"] = np.where(df_trade["signal"] > 2, -1, df_trade["position"])  # Short
    
    # Returns
    df_trade["price_return"] = df_trade.groupby("coupon_bucket")["price"].pct_change()
    df_trade["pnl"] = df_trade["position"].shift(1) * df_trade["price_return"]
    df_trade["cum_pnl"] = df_trade.groupby("coupon_bucket")["pnl"].cumsum()
    
    # Stats
    total_trades = (df_trade["position"] != 0).sum()
    
    if total_trades > 0:
        winning_trades = (df_trade["pnl"] > 0).sum()
        total_pnl = df_trade["pnl"].sum()
        avg_pnl = df_trade["pnl"].mean()
        sharpe = (df_trade["pnl"].mean() / df_trade["pnl"].std() * np.sqrt(12)) if df_trade["pnl"].std() > 0 else 0
        
        print("\n  ═══════════════════════════════════════════")
        print("  TRADING SIMULATION RESULTS")
        print("  ═══════════════════════════════════════════")
        print(f"  Strategy: Long undervalued coupons\n")
        print(f"  Total Trades: {total_trades}")
        print(f"  Winning Trades: {winning_trades} ({winning_trades/total_trades*100:.1f}%)")
        print(f"  Total P&L: {total_pnl:.4f}")
        print(f"  Avg P&L/Trade: {avg_pnl:.6f}")
        print(f"  Sharpe (annualized): {sharpe:.2f}")
        print("  ═══════════════════════════════════════════\n")
        
        trade_path = PROCESSED_DIR / "trading_results_2022_2023.csv"
        df_trade.to_csv(trade_path, index=False)
        print(f"  ✓ Saved to {trade_path}")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nKey Outputs:")
print(f"  • {PROCESSED_DIR / 'panel_consolidated_2022_2023.parquet'}")
print(f"  • {PROCESSED_DIR / 'feature_importance_logit.csv'}")
print(f"  • {PROCESSED_DIR / 'cpr_by_month_coupon_2022_2023.csv'}")