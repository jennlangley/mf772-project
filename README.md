# Mortgage Prepayment Modeling and Trading Pipeline

This project builds a loan-level and pool-level prepayment model using Fannie Mae Single-Family Loan Performance and Acquisition data (2022–2023).  
The objective is to estimate monthly prepayment probability (SMM), aggregate it to pool-level CPR, compare model CPR to market-implied CPR from FNCL prices, and evaluate a relative-value trading rule.

---

## 1. Data Overview

**Source:** Fannie Mae Single-Family Loan Performance Dataset  
**Period:** 2022 Q1 – 2023 Q4  
**Approx. Rows After Sampling:** ~2.8M  
**Final Panel Features:** Loan characteristics, incentive variables, burnout measures, seasonality, geography, and interaction terms.

Two final datasets are produced:

- `panel_consolidated_2022_2023.parquet` — loan-level data with engineered features  
- `cpr_by_month_coupon_2022_2023.csv` — pool-level model and actual CPR for all month/coupon combinations  

---

## 2. Modeling Approach

### Target Definition
- **Loan-level target:**  
  `y_next = 1` if loan prepays in the next month, else `0`.

### Model
- **Logistic Regression**
  - Robust under extreme class imbalance  
  - Interpretable coefficients  
  - Typical loan-level AUC of ~0.62–0.63 (consistent with literature)

### Key Features
- Refinance incentive (`refi_incentive`, sigmoid and interaction terms)  
- Burnout measures  
- Original and current LTV  
- Rate incentive  
- Loan age  
- Geography (state)  
- Purpose, channel, FICO interactions  

---

## 3. Pool-Level CPR Estimation

Loan-level predicted SMM is aggregated into:
- **SMM per coupon-month pool**
- **CPR per coupon-month pool**

Outputs include:
- Model CPR (`cpr_model`)
- Actual CPR (`cpr_actual`)
- Residual diagnostics

---

## 4. Trading Framework

Market-implied CPR is derived from FNCL TBA prices.  
Strategy:

- **Long undervalued coupons** (model CPR < market-implied CPR)  
- **Short overvalued coupons** (model CPR > market CPR)

Performance metrics:
- Win rate  
- Cumulative P&L  
- Monthly P&L  
- Rolling Sharpe ratio  

---

## 5. Visualization

`plot_results.py` generates a set of diagnostic and performance graphs:

- CPR model vs actual (aggregate + scatter)
- Heatmaps by coupon bucket (model / actual / residual)
- CPR error distribution
- ROC, PR, and calibration curves
- Logistic regression feature importance
- Loan-level descriptive plots  
- Trading cumulative P&L  
- Monthly P&L  
- PNL distribution  
- Rolling Sharpe ratio  

All graphs save to:

/graphs


---

## 6. File Structure

project/
│
├── data/
│ └── processed/
│ ├── panel_consolidated_2022_2023.parquet
│ ├── cpr_by_month_coupon_2022_2023.csv
│ ├── cpr_vs_fncl_2022_2023.csv
│ ├── trading_results_2022_2023.csv
│ └── feature_importance_logit.csv
│
├── new.py # Main modeling + CPR + trading pipeline
├── plot_results.py # Graph generation script
└── graphs/ # Generated figures


---

## 7. Requirements

Python 3.10+  
Dependencies:

pandas
numpy
scikit-learn
seaborn
matplotlib
pyarrow


---

## 8. How to Run

**Run full pipeline:**

python new.py


**Generate all graphs:**

python plot_results.py


Outputs appear in:
- `data/processed/`
- `graphs/`

---