from app.ml.inference.predict import ChurnPredictionPipeline


class ChurnPredictionConsole:

    def __init__(self):
        self.pipeline = ChurnPredictionPipeline()

        self.sample = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 69.50,
            "TotalCharges": 4170.00,
        }

    def predict(self):
        result = self.pipeline.predict(self.sample)

        print("=" * 60)
        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']:.2%}")
        print("=" * 60)

        print("\nProbabilities")
        print("-" * 60)
        print(f"No Churn : {result['probabilities']['No Churn']:.2%}")
        print(f"Churn    : {result['probabilities']['Churn']:.2%}")

        print("=" * 60)


if __name__ == "__main__":
    ChurnPredictionConsole().predict()