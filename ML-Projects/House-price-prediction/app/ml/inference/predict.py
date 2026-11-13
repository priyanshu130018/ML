from pathlib import Path

import pandas as pd

from app.ml.inference.model_loader import HouseModelLoader


class HousePricePredictor:

    def __init__(self, model_dir: Path | None = None):
        loader = HouseModelLoader(model_dir=model_dir)
        self.model, self.scaler = loader.load()

    def predict(self, features: dict):
        house = pd.DataFrame([features])

        if hasattr(self.model, "feature_names_in_"):
            house = house[list(self.model.feature_names_in_)]

        house_scaled = self.scaler.transform(house)
        prediction = self.model.predict(house_scaled)

        return float(prediction[0])