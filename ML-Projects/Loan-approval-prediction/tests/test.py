from sklearn.tree import export_text
import joblib
import pandas as pd
from app.ml.preprocessing.feature_engineering import FeatureEngineer

pipeline = joblib.load("saved_models/loan_approval_model.pkl")

classifier = pipeline.named_steps["classifier"]

classifier = pipeline.named_steps["classifier"]

#print("Classes:", classifier.classes_)

feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

#print(export_text(classifier, feature_names=list(feature_names)))



df = pd.read_csv("data/processed/eda_checked.csv")

sample = df.iloc[[0]]

print("Actual label:")
print(sample["loan_status"])

X = sample.drop(columns=["loan_status"])

# Apply feature engineering
X = FeatureEngineer().transform(X)

print("Prediction :", pipeline.predict(X))
print("Probability:", pipeline.predict_proba(X))