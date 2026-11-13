import pandas as pd

from app.ml.inference.model_loader import ModelLoader
from app.ml.inference.predictor import Predictor
from app.ml.inference.postprocessor import PostProcessor


class PredictionPipeline:

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        predictor: Predictor | None = None,
        postprocessor: PostProcessor | None = None,
    ):
        self.model_loader = model_loader or ModelLoader()
        self.model = self.model_loader.load("fraud_detection_model")
        self.predictor = predictor or Predictor(self.model)
        self.postprocessor = postprocessor or PostProcessor()

    def predict(self, data: dict | pd.DataFrame) -> dict:

        prediction, probability = self.predictor.predict(data)

        result = self.postprocessor.process(
            prediction,
            probability,
        )

        print("=" * 60)
        print("FRAUD DETECTION PREDICTION")
        print("=" * 60)

        for key, value in result.items():
            print(f"{key}: {value}")

        print("=" * 60)

        return result