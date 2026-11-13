import pandas as pd


class ChurnDataCleaner:
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()

        if "customerID" in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns="customerID")

        cleaned_df["TotalCharges"] = pd.to_numeric(
            cleaned_df["TotalCharges"],
            errors="coerce",
        )

        return cleaned_df.dropna()
