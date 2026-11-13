import numpy as np


class DataValidator:

    def __init__(self, target_column):
        self.target_column = target_column

    def validate(self, df):
        if df.empty:
            raise ValueError("Dataset is empty.")

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names found.")

        if df.duplicated().any():
            raise ValueError("Duplicate rows found.")

        if df.isnull().sum().sum() > 0:
            raise ValueError("Dataset contains missing values.")

        numeric_df = df.select_dtypes(include=[np.number])

        if np.isinf(numeric_df.to_numpy()).any():
            raise ValueError("Dataset contains infinite values.")


        if df[self.target_column].nunique() != 2:
            raise ValueError("Target column must be binary.")

        return True