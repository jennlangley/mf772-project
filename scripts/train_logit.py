from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "panel_2023Q1_small.parquet"

df = pd.read_parquet(PANEL)
print("Panel shape:", df.shape)
print(df.head())

num_features = [
    "age_m", "season_sin", "season_cos",
    "fico", "ltv", "dti", "orig_rate", "orig_term",
]

cat_features = ["state", "purpose", "channel"]
cat_features = [c for c in cat_features if c in df.columns]

X = df[num_features + cat_features]
y = df["y_next"]

print("Positive event rate (prepay next month):", y.mean())

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_features),
    ]
)

clf = Pipeline(
    steps=[
        ("prep", preprocess),
        ("logit", LogisticRegression(max_iter=500, n_jobs=-1)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)

y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
print("Logistic regression AUC:", round(auc, 4))
print("\nClassification report (threshold 0.5):")
print(classification_report(y_test, y_pred))
