from sklearn.linear_model import LogisticRegression


class ChurnModelProvider:
    
    def get_model(self):
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
