from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.inspection import permutation_importance

from app.ml.data.data_loader import DataLoader
from app.config.settings import TARGET_COLUMN

BASE_DIR = Path(__file__).resolve().parents[2]

model_path = Path("saved_models/fraud_detection_model.pkl")

pipeline = load(model_path)

# Load original dataset
df = DataLoader().load()

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X = X.sample(50000, random_state=42)
y = y.loc[X.index]

result = permutation_importance(
    pipeline,
    X,
    y,
    n_repeats=10,
    random_state=42,
    scoring="accuracy"
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)