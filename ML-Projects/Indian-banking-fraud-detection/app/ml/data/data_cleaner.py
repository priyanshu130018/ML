import numpy as np
import pandas as pd


class DataCleaner:

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df.drop_duplicates(inplace=True)

        df.drop(
            columns=[
                "transaction_id",
                "customer_id",
                "has_loan",
                "loan_type",
                "transaction_status",
            ],
            inplace=True,
            errors="ignore",
        )

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns

        categorical_columns = df.select_dtypes(
            exclude=["number"]
        ).columns

        df[numeric_columns] = df[numeric_columns].fillna(
            df[numeric_columns].median()
        )

        df[categorical_columns] = df[
            categorical_columns
        ].fillna(
            df[categorical_columns].mode().iloc[0]
        )

        return df