import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if "transaction_amount" in df.columns:

            df["log_transaction_amount"] = np.log1p(
                df["transaction_amount"]
            )

            df["high_amount_transaction"] = (
                df["transaction_amount"] > 100000
            ).astype(int)

        if "transaction_date" in df.columns:

            df["transaction_date"] = pd.to_datetime(
                df["transaction_date"]
            )


            df["transaction_day"] = (
                df["transaction_date"].dt.day
            )

            df["transaction_month"] = (
                df["transaction_date"].dt.month
            )

            df["transaction_weekday"] = (
                df["transaction_date"].dt.weekday
            )

            df["is_weekend"] = (
                df["transaction_weekday"] >= 5
            ).astype(int)

            df.drop(
                columns=["transaction_date"],
                inplace=True,
            )

        return df