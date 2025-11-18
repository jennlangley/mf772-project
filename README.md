# Mortgage Prepayment Modeling & FNCL Relative Value  
**MF772 – Credit Risk Project (2025)**  

This project builds a monthly mortgage **prepayment model** using Fannie Mae Single-Family Loan Performance data, then links model-implied **CPR/SMM** to **FNCL TBA price data** for coupon-level relative value analysis.

## Project Structure
```
mf772-project/
├── data/
│   ├── processed/
│   │   ├── panel_2023Q1_small.parquet
│   │   ├── cpr_by_month_coupon_2023Q1.csv
│   │   └── cpr_vs_fncl_2023Q1.csv
│   └── FNCL_price_data.csv
├── models/
│   ├── xgb_prepay_2023Q1_gpu.json
│   ├── xgb_feature_importance_2023Q1.csv
│   └── xgb_features_2023Q1.txt
├── scripts/
│   ├── build_panel_2023.py
│   ├── train_logit.py
│   ├── train_xgb_gpu.py
│   ├── score_and_aggregate_cpr.py
│   └── merge_cpr_fncl.py
└── README.md
```

## Pipeline Overview
### 1. Build the Loan-Month Panel
Convert parquet-min data into a clean model-ready dataset with:
- loan age  
- seasonality  
- fico/ltv/dti  
- prepayment label (`y_next`)

Run:
```bash
python scripts/build_panel_2023.py
```

### 2. Train Models
**Logistic Regression:**
```bash
python scripts/train_logit.py
```

**GPU XGBoost:**
```bash
python scripts/train_xgb_gpu.py
```

### 3. Generate CPR
Convert predicted SMM to CPR by coupon + month:
```bash
python scripts/score_and_aggregate_cpr.py
```

### 4. Merge CPR with FNCL Prices
```bash
python scripts/merge_cpr_fncl.py
```

## Requirements
- Python 3.10+
- pandas, pyarrow, numpy
- scikit-learn
- xgboost (GPU optional)

## Notes
- Raw data is excluded due to size.
- The repository contains only the small processed panel and model outputs.
