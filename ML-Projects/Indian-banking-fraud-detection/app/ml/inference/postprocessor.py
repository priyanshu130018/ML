class PostProcessor:

    def process(self, prediction, probability):
        return {
            "prediction": (
                "Fraud Transaction"
                if prediction == 1
                else "Legitimate Transaction"
            ),
            "confidence": float(probability[prediction]),
            "fraud_probability": float(probability[1]),
            "legitimate_probability": float(probability[0]),
        }