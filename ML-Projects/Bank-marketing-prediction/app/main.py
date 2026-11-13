from app.ml.pipelines.prediction_pipeline import PredictionPipeline


class BankMarketingPredictionScript:

    def __init__(self):
        self.pipeline = PredictionPipeline()

        self.sample = {
            "age": 35,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 1500,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "may",
            "duration": 250,
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }

    def run(self):
        self.pipeline.predict(self.sample)


if __name__ == "__main__":
    BankMarketingPredictionScript().run()