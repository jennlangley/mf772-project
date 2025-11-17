from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "panel_2023Q1_small.parquet"

df = pd.read_parquet(PANEL)
print("Panel shape:", df.shape)

num_features = [
    "age_m", "season_sin", "season_cos",
    "fico", "ltv", "dti", "orig_rate", "orig_term",
]

cat_features = ["state", "purpose", "channel"]
cat_features = [c for c in cat_features if c in df.columns]

X = df[num_features + cat_features].copy()
y = df["y_next"]

# one-hot encode cats with pandas
if cat_features:
    X = pd.get_dummies(X, columns=cat_features, dummy_na=False)

print("Features after dummies:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# choose tree_method/device depending on xgboost version
# try hist + device="cuda" (xgboost>=2.0); if it errors, switch to tree_method="gpu_hist"
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    tree_method="hist",
    device="cuda",    # if this errors, change to: tree_method="gpu_hist" and remove device
    eval_metric="auc",
    n_jobs=4,
)

eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=50,
    early_stopping_rounds=50,
)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("XGBoost AUC:", round(auc, 4))

# save model + feature importances
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "xgb_prepay_2023Q1_gpu.json"
model.save_model(model_path)
print("Saved model to", model_path)

fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

fi_path = MODEL_DIR / "xgb_feature_importance_2023Q1.csv"
fi.to_csv(fi_path, index=False)
print("Saved feature importances to", fi_path)
print(fi.head(20))
