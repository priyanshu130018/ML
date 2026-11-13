import pandas as pd

from sklearn.pipeline import Pipeline


class Predictor:

    def __init__(self, model: Pipeline):
        self.model = model

    def predict(self, data: dict):

        if isinstance(data, dict):
            data = pd.DataFrame([data])

        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0]

        return prediction, probability