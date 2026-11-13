import pandas as pd

from app.ml.inference.model_loader import ModelLoader
from app.ml.inference.predict import Predictor
from app.ml.inference.postprocessing import PostProcessor


class PredictionPipeline:

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        predictor: Predictor | None = None,
        post_processor: PostProcessor | None = None,
    ):
        self.model_loader = model_loader or ModelLoader()
        self.model = self.model_loader.load("loan_approval_model")
        self.predictor = predictor or Predictor(self.model)
        self.post_processor = post_processor or PostProcessor()

    def predict(self, sample: dict) -> dict:
        df = pd.DataFrame([sample])

        prediction, probability = self.predictor.predict(df)

        return self.post_processor.process(
            prediction,
            probability,
        )