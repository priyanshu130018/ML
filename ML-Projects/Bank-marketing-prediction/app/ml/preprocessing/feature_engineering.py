import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["high_balance"] = (
            X["balance"] > 50000
        ).astype(int)

        X["long_call"] = (
            X["duration"] > 300
        ).astype(int)

        X["multiple_contacts"] = (
            X["campaign"] > 3
        ).astype(int)

        X["previously_contacted"] = (
            X["pdays"] != -1
        ).astype(int)

        X["has_previous_contact"] = (
            X["previous"] > 0
        ).astype(int)

        X["balance_per_age"] = (
            X["balance"] / (X["age"] + 1)
        )

        X["contacts_per_day"] = (
            X["campaign"] / (X["day"] + 1)
        )

        X["previous_success"] = np.where(
            X["poutcome"] == "success",
            1,
            0
        )

        return X