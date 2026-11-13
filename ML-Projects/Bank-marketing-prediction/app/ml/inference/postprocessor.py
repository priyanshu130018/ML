class PostProcessor:

    def process(self, prediction, probability):

        return {
            "prediction": (
                "Deposit: Yes"
                if prediction == 1
                else "Deposit: No"
            ),
            "confidence": f"{probability[prediction]:.2%}",
            "probability_yes": f"{probability[1]:.2%}",
            "probability_no": f"{probability[0]:.2%}",
        }