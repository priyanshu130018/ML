from app.ml.pipelines.prediction_pipeline import PredictionPipeline


class LoanPredictionScript:

    def __init__(self):
        self.pipeline = PredictionPipeline()

        self.sample = {
            "person_age": 26,
            "person_gender": "female",
            "person_education": "Bachelor",
            "person_income": 8.0,
            "person_emp_exp": 1,
            "person_home_ownership": "MORTGAGE",
            "loan_amnt": 10000.0,
            "loan_intent": "DEBTCONSOLIDATION",
            "loan_int_rate": 15.99,
            "loan_percent_income": 1,
            "cb_person_cred_hist_length": 3.0,
            "credit_score": 63,
            "previous_loan_defaults_on_file": "No",
        }

    def predict(self):
        result = self.pipeline.predict(self.sample)
        
        print("=" * 60)
        print("LOAN APPROVAL PREDICTION")
        print("=" * 60)

        print(f"Prediction : {result['prediction']}")
        print(f"Confidence : {result['confidence']:.2%}")
        print("=" * 60)

        print()
        print("-" * 60)
        print("Probabilities")
        print("-" * 60)
        
        print(f"Rejected : {result['probabilities']['Rejected']:.2%}")
        print(f"Approved : {result['probabilities']['Approved']:.2%}")
    
        print("=" * 60)
        

if __name__ == "__main__":
    LoanPredictionScript().predict()
