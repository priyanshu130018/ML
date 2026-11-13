import pandas as pd


class DataValidation:

    REQUIRED_COLUMNS = [
        "person_age",
        "person_gender",
        "person_education",
        "person_income",
        "person_emp_exp",
        "person_home_ownership",
        "loan_amnt",
        "loan_intent",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
        "previous_loan_defaults_on_file",
        "loan_status",
    ]

    def validate(self, df: pd.DataFrame) -> bool:

        if df.empty:
            raise ValueError("Dataset is empty.")

        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        if not set(df["loan_status"].unique()).issubset({0, 1}):
            raise ValueError("Target column must contain only 0 and 1.")

        return True