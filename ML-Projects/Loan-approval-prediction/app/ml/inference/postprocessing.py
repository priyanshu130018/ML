class PostProcessor:

    def process(self, prediction, probability):
        rejected_prob, approved_prob = probability[0]

        label = "Approved" if int(prediction[0]) == 1 else "Rejected"
        
        return {
            "prediction": label,
            "confidence": max(rejected_prob, approved_prob),

            "probabilities": {
                "Rejected": float(rejected_prob),
                "Approved": float(approved_prob),
            },
        }