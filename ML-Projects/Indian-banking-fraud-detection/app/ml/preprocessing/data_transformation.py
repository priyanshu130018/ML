from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


class DataTransformer:

    def build_pipeline(self, X):

        numeric_features = X.select_dtypes(
            include=[
                "int64",
                "float64",
                "bool",
            ]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features,
                ),
            ]
        )