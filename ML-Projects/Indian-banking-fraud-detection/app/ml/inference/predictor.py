import pandas as pd


class Predictor:

    def __init__(self, model):
        self.model = model

    def predict(self, data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0]

        return prediction, probability