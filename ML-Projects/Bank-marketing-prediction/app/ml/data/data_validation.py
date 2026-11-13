import pandas as pd

from app.config.settings import TARGET_COLUMN


class DataValidation:

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Dataset is empty.")

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' not found."
            )

        if df.duplicated().any():
            df = df.drop_duplicates().reset_index(drop=True)

        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

        if df[TARGET_COLUMN].isnull().any():
            raise ValueError(
                "Target column contains invalid values."
            )

        return df