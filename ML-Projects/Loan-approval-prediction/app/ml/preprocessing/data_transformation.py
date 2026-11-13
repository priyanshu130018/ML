from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:

    def __init__(self):
        self.numeric_features = [
            "person_age",
            "person_income",
            "person_emp_exp",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "income_per_year_exp",
            "loan_income_ratio",
            "credit_age_ratio",
            "income_age_ratio",
        ]

        self.categorical_features = [
            "person_gender",
            "person_education",
            "person_home_ownership",
            "loan_intent",
            "previous_loan_defaults_on_file",
        ]

    def build_pipeline(self):
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )