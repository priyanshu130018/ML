import pandas as pd
from sklearn.datasets import fetch_california_housing


class HousingDataLoader:

    def load(self) -> pd.DataFrame:
        housing = fetch_california_housing(as_frame=True)
        return housing.frame
