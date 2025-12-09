import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

# ---------------------------------------
# Setup
# ---------------------------------------
BASE = Path(".")
DATA_DIR = BASE / "data" / "processed"
GRAPH_DIR = BASE / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

# Load data
panel_path = DATA_DIR / "panel_consolidated_2022_2023.parquet"
df_panel = pd.read_parquet(panel_path)

df_cpr = pd.read_csv(DATA_DIR / "cpr_by_month_coupon_2022_2023.csv")
df_trade = pd.read_csv(DATA_DIR / "trading_results_2022_2023.csv")
df_fi = pd.read_csv(DATA_DIR / "feature_importance_logit.csv")

# ---------------------------------------
# Helper: save + close
# ---------------------------------------
def savefig(name):
    out = GRAPH_DIR / f"{name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

sns.set(style="whitegrid")

# ---------------------------------------
# Time conversions
# ---------------------------------------
# CPR & trading files have year_month string like '2022-01'
df_cpr["year_month"] = pd.to_datetime(df_cpr["year_month"])
df_trade["year_month"] = pd.to_datetime(df_trade["year_month"])

if "year_month" in df_panel.columns:
    df_panel["year_month"] = pd.to_datetime(df_panel["year_month"])
if "act_period_dt" in df_panel.columns:
    df_panel["act_period_dt"] = pd.to_datetime(df_panel["act_period_dt"])

# --------------------------------------------------------------------
# A. LOAN-LEVEL DESCRIPTIVE GRAPHS (panel file, actual y_next)
# --------------------------------------------------------------------
# These do NOT use predictions, only actual prepayment indicator y_next

y_col = "y_next"
if y_col not in df_panel.columns:
    raise ValueError(f"{y_col} not found in panel_consolidated_2022_2023.parquet")

df_panel = df_panel.dropna(subset=[y_col])

# A1. Prepayment rate by loan age
if "age_m" in df_panel.columns:
    df_age = (
        df_panel.groupby("age_m")[y_col]
        .mean()
        .reset_index()
        .sort_values("age_m")
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_age, x="age_m", y=y_col)
    plt.title("Prepayment Rate by Loan Age")
    plt.xlabel("Loan Age (months)")
    plt.ylabel("Prepayment Rate (next month)")
    savefig("loan_prepay_rate_by_age")

# A2. Prepayment rate by original LTV decile
if "ltv" in df_panel.columns:
    df_ltv = df_panel.copy()
    df_ltv = df_ltv.dropna(subset=["ltv"])
    df_ltv["ltv_decile"] = pd.qcut(df_ltv["ltv"], 10, labels=False)

    df_ltv_rate = (
        df_ltv.groupby("ltv_decile")[y_col]
        .mean()
        .reset_index()
        .sort_values("ltv_decile")
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_ltv_rate, x="ltv_decile", y=y_col)
    plt.title("Prepayment Rate by Original LTV Decile")
    plt.xlabel("Original LTV Decile (0 = lowest, 9 = highest)")
    plt.ylabel("Prepayment Rate (next month)")
    savefig("loan_prepay_rate_by_ltv_decile")

# A3. Prepayment rate by refinancing incentive bin
if "refi_incentive" in df_panel.columns:
    df_inc = df_panel.copy()
    df_inc = df_inc.dropna(subset=["refi_incentive"])
    df_inc["inc_bin"] = pd.qcut(df_inc["refi_incentive"], 10, labels=False)

    df_inc_rate = (
        df_inc.groupby("inc_bin")[y_col]
        .mean()
        .reset_index()
        .sort_values("inc_bin")
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_inc_rate, x="inc_bin", y=y_col)
    plt.title("Prepayment Rate by Refi Incentive Decile")
    plt.xlabel("Refi Incentive Decile (0 = lowest, 9 = highest)")
    plt.ylabel("Prepayment Rate (next month)")
    savefig("loan_prepay_rate_by_refi_incentive_decile")

# --------------------------------------------------------------------
# B. POOL-LEVEL MODEL DIAGNOSTICS (CPR file)
# --------------------------------------------------------------------
# Here we DO have model outputs: smm_model, smm_actual, cpr_model, cpr_actual

required_cpr_cols = ["smm_model", "smm_actual", "cpr_model", "cpr_actual"]
missing = [c for c in required_cpr_cols if c not in df_cpr.columns]
if missing:
    raise ValueError(f"Missing columns in cpr_by_month_coupon_2022_2023.csv: {missing}")

# B1. CPR model vs actual over time (aggregated across coupons)
df_cpr_agg = (
    df_cpr.groupby("year_month")[["cpr_model", "cpr_actual"]]
    .mean()
    .reset_index()
    .sort_values("year_month")
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_cpr_agg, x="year_month", y="cpr_model", label="Model CPR")
sns.lineplot(data=df_cpr_agg, x="year_month", y="cpr_actual", label="Actual CPR")
plt.title("CPR – Model vs Actual (Average Across Coupons)")
plt.xlabel("Year-Month")
plt.ylabel("CPR")
plt.xticks(rotation=45)
plt.legend()
savefig("cpr_model_vs_actual_agg")

# B2. Scatter: CPR actual vs model (pool-level)
plt.figure(figsize=(7, 7))
sns.scatterplot(x=df_cpr["cpr_actual"], y=df_cpr["cpr_model"], alpha=0.5)
lo = min(df_cpr["cpr_actual"].min(), df_cpr["cpr_model"].min())
hi = max(df_cpr["cpr_actual"].max(), df_cpr["cpr_model"].max())
plt.plot([lo, hi], [lo, hi], "r--")
plt.title("Pool-Level CPR – Actual vs Model")
plt.xlabel("Actual CPR")
plt.ylabel("Model CPR")
savefig("cpr_scatter_pool")

# B3. CPR error distribution (pool-level)
df_cpr["cpr_error"] = df_cpr["cpr_model"] - df_cpr["cpr_actual"]

plt.figure(figsize=(10, 5))
sns.histplot(df_cpr["cpr_error"], bins=40, kde=True)
plt.title("Distribution of CPR Prediction Error (Model − Actual)")
plt.xlabel("CPR Error")
plt.ylabel("Count")
savefig("cpr_error_distribution")

# B4. Heatmaps by coupon bucket – model, actual, residuals
pivot_model = df_cpr.pivot(
    index="year_month", columns="coupon_bucket", values="cpr_model"
)
pivot_actual = df_cpr.pivot(
    index="year_month", columns="coupon_bucket", values="cpr_actual"
)
pivot_resid = df_cpr.pivot(
    index="year_month", columns="coupon_bucket", values="cpr_error"
)

plt.figure(figsize=(12, 7))
sns.heatmap(pivot_model, cmap="Blues")
plt.title("CPR (Model) by Month and Coupon Bucket")
plt.xlabel("Coupon Bucket")
plt.ylabel("Year-Month")
plt.xticks(rotation=45)
savefig("heatmap_cpr_model")

plt.figure(figsize=(12, 7))
sns.heatmap(pivot_actual, cmap="Reds")
plt.title("CPR (Actual) by Month and Coupon Bucket")
plt.xlabel("Coupon Bucket")
plt.ylabel("Year-Month")
plt.xticks(rotation=45)
savefig("heatmap_cpr_actual")

plt.figure(figsize=(12, 7))
sns.heatmap(pivot_resid, cmap="coolwarm", center=0)
plt.title("CPR Residuals (Model − Actual) by Month and Coupon")
plt.xlabel("Coupon Bucket")
plt.ylabel("Year-Month")
plt.xticks(rotation=45)
savefig("heatmap_cpr_residuals")

# B5. Pool-level ROC / PR / Calibration using SMM
# Define a binary event at pool level: "any prepay" vs "none"
y_true_pool = (df_cpr["smm_actual"] > 0).astype(int).values
y_score_pool = df_cpr["smm_model"].astype(float).values

# ROC
fpr, tpr, _ = roc_curve(y_true_pool, y_score_pool)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve – Pool-Level Prepayment (Any Prepay vs None)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
savefig("roc_curve_pool")

# Precision–Recall
prec, rec, _ = precision_recall_curve(y_true_pool, y_score_pool)

plt.figure(figsize=(7, 6))
plt.plot(rec, prec)
plt.title("Precision–Recall Curve – Pool-Level Prepayment")
plt.xlabel("Recall")
plt.ylabel("Precision")
savefig("pr_curve_pool")

# Calibration
prob_true, prob_pred = calibration_curve(y_true_pool, y_score_pool, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "k--")
plt.title("Calibration Curve – Pool-Level (Predicted vs Actual Frequency)")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
savefig("calibration_curve_pool")

# --------------------------------------------------------------------
# C. FEATURE IMPORTANCE (LOGISTIC REGRESSION)
# --------------------------------------------------------------------
if "coefficient" not in df_fi.columns or "feature" not in df_fi.columns:
    raise ValueError("feature_importance_logit.csv must have 'feature' and 'coefficient'.")

df_fi_sorted = df_fi.copy()
df_fi_sorted["abs_coef"] = df_fi_sorted["coefficient"].abs()
df_fi_sorted = df_fi_sorted.sort_values("abs_coef", ascending=False)

top_n = 25
plt.figure(figsize=(8, 10))
sns.barplot(
    data=df_fi_sorted.head(top_n),
    x="coefficient",
    y="feature",
)
plt.title("Top 25 Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
savefig("logit_feature_importance")

# --------------------------------------------------------------------
# D. TRADING PERFORMANCE GRAPHS
# --------------------------------------------------------------------
# D1. Cumulative P&L
df_trade_sorted = df_trade.sort_values("year_month").copy()
df_trade_sorted["pnl_filled"] = df_trade_sorted["pnl"].fillna(0)
df_trade_sorted["cum_pnl"] = df_trade_sorted["pnl_filled"].cumsum()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_trade_sorted, x="year_month", y="cum_pnl")
plt.title("Trading Strategy – Cumulative P&L")
plt.xlabel("Year-Month")
plt.ylabel("Cumulative P&L")
plt.xticks(rotation=45)
savefig("trading_cum_pnl")

# D2. Monthly P&L
monthly_pnl = (
    df_trade_sorted.groupby(df_trade_sorted["year_month"].dt.to_period("M"))["pnl_filled"]
    .sum()
    .reset_index()
)
monthly_pnl["year_month"] = monthly_pnl["year_month"].dt.to_timestamp()

plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_pnl, x="year_month", y="pnl_filled")
plt.title("Monthly P&L")
plt.xlabel("Year-Month")
plt.ylabel("P&L")
plt.xticks(rotation=45)
savefig("monthly_pnl")

# D3. Trade-level P&L distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_trade_sorted["pnl_filled"], bins=40, kde=True)
plt.title("Distribution of Trade-Level P&L")
plt.xlabel("P&L")
plt.ylabel("Count")
savefig("pnl_distribution")

# D4. Rolling Sharpe ratio (20-trade window)
roll_mean = df_trade_sorted["pnl_filled"].rolling(20).mean()
roll_std = df_trade_sorted["pnl_filled"].rolling(20).std()
df_trade_sorted["rolling_sharpe"] = roll_mean / roll_std

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_trade_sorted, x="year_month", y="rolling_sharpe")
plt.title("Rolling 20-Trade Sharpe Ratio")
plt.xlabel("Year-Month")
plt.ylabel("Rolling Sharpe")
plt.xticks(rotation=45)
savefig("rolling_sharpe_20trades")

print("\nAll graphs saved successfully to ./graphs")
