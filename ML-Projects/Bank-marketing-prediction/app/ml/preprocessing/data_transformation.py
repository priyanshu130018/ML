import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataTransformer:

    def build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ])

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_columns,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                ),
            ]
        )

        return preprocessor