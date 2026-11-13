from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HousingModelEvaluator:

    def evaluate(self, model, X_test, y_test):
        
        predictions = model.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        }