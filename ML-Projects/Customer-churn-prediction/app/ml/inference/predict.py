import pandas as pd

from app.ml.data.data_cleaning import ChurnDataCleaner
from app.ml.inference.model_loader import ChurnModelLoader


class ChurnPredictionPipeline:

    def __init__(
        self,
        model_loader: ChurnModelLoader | None = None,
        cleaner: ChurnDataCleaner | None = None,
    ):
        self.model_loader = model_loader or ChurnModelLoader()
        self.cleaner = cleaner or ChurnDataCleaner()

        self.model = self.model_loader.load(
            "churn_prediction_model"
        )

    def predict(self, sample: dict) -> dict:
        df = pd.DataFrame([sample])

        df = self.cleaner.clean(df)

        prediction = int(self.model.predict(df)[0])
        probability = self.model.predict_proba(df)[0]

        label = "Churn" if prediction == 1 else "No Churn"

        probabilities = {
            "No Churn": float(probability[0]),
            "Churn": float(probability[1]),
        }

        confidence = float(probability[prediction])

        return {
            "prediction": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }