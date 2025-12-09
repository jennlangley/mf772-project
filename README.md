# Mortgage Prepayment Modeling and Trading Pipeline

This project builds a loan-level and pool-level prepayment model using Fannie Mae Single-Family Loan Performance and Acquisition data (2022–2023). The objective is to estimate monthly prepayment probability (SMM), aggregate it to pool-level CPR, compare model CPR to market‑implied CPR from FNCL prices, and evaluate a relative‑value trading rule.

## 1. Data Overview
• Source: Fannie Mae Single‑Family Loan Performance Dataset  
• Period: 2022 Q1 – 2023 Q4  
• Rows after sampling: ~2.8M  
• Features: loan characteristics, incentives, burnout, seasonality, geography, interactions

Main processed outputs:
- panel_consolidated_2022_2023.parquet  
- cpr_by_month_coupon_2022_2023.csv  

## 2. Modeling Approach
### Target
Loans labeled `1` if they prepay next month, otherwise `0`.

### Model: Logistic Regression
Chosen for:
- robustness under extreme imbalance  
- interpretability  
- stable AUC in the 0.62–0.63 range (industry‑typical)

### Major Predictors
- refinance incentive + sigmoids + interactions  
- burnout factor  
- current/original LTV  
- rate incentive  
- loan age  
- geography  
- purpose, channel, FICO interactions  

## 3. CPR Aggregation
Loan‑level SMM predictions → grouped into coupon‑month buckets → CPR using:
CPR = 1 − (1 − SMM)¹²

Both model CPR and actual CPR are computed.

## 4. Trading Strategy
Market‑implied CPR comes from FNCL TBA prices.  
Trading rule:
- Long when model CPR < market CPR  
- Short when model CPR > market CPR  

Outputs:
- win rate  
- cumulative P&L  
- monthly P&L  
- rolling Sharpe ratio  

## 5. Visualization
`plot_results.py` generates:
- CPR model vs actual  
- scatter + error distribution  
- heatmaps (model, actual, residual)  
- ROC, PR, calibration curves  
- logistic feature importance  
- trading P&L diagnostics  

Figures saved to `graphs/`.

## 6. Directory Structure
project/
├── data/processed/  
├── new.py  
├── plot_results.py  
└── graphs/

## 7. Requirements
Python 3.10+  
Packages: pandas, numpy, sklearn, seaborn, matplotlib, pyarrow

## 8. How to Run
Pipeline:
    python new.py
Graphs:
    python plot_results.py
