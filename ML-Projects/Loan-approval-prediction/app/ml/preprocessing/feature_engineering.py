import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        data["income_per_year_exp"] = (
            data["person_income"] /
            (data["person_emp_exp"] + 1)
        )

        data["loan_income_ratio"] = (
            data["loan_amnt"] /
            data["person_income"]
        )

        data["credit_age_ratio"] = (
            data["credit_score"] /
            (data["cb_person_cred_hist_length"] + 1)
        )

        data["income_age_ratio"] = (
            data["person_income"] /
            data["person_age"]
        )

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        return data