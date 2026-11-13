from sklearn.pipeline import Pipeline


class Predictor:

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict(self, df):
        prediction = self.pipeline.predict(df)
        probability = self.pipeline.predict_proba(df)

        return prediction, probability