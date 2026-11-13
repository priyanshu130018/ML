from app.ml.pipelines.prediction_pipeline import PredictionPipeline


class FraudPredictionScript:

    def __init__(self):
        self.pipeline = PredictionPipeline()

        self.sample = {
            "transaction_amount": 250000,
            "transaction_type": "UPI",
            "merchant_category": "Electronics",
            "account_balance": 500000,
            "transaction_date": "2025-08-10 14:35:00",
            "transaction_time": "14:35",
            "transaction_hour": 14,
            "channel": "Mobile Banking",
            "state": "Maharashtra",
            "account_type": "Savings",
            "kyc_status": "Verified",
            "loan_type": "Home Loan",
            "has_loan": True,
            "credit_score": 780,
            "emi_amount": 25000,
            "transaction_status": "Completed",
            "transaction_direction": "Debit",
        }

    def run(self):
        self.pipeline.predict(self.sample)


if __name__ == "__main__":
    FraudPredictionScript().run()