from app.ml.training.evaluate import HousingModelEvaluator

class HousingModelComparer:

    def __init__(self, evaluator: HousingModelEvaluator | None = None):
        self.evaluator = evaluator or HousingModelEvaluator()

    def compare(self, models, X_test, y_test):
        results = {}

        for name, model in models.items():
            results[name] = self.evaluator.evaluate(
                model,
                X_test,
                y_test,
            )

        best_mae = min(results, key=lambda x: results[x]["MAE"])
        best_rmse = min(results, key=lambda x: results[x]["RMSE"])
        best_r2 = max(results, key=lambda x: results[x]["R2"])

        return {
            "results": results,
            "best_model": models[best_r2],
            "best_name": best_r2,
            "best_score": results[best_r2]["R2"],
            "best_mae": best_mae,
            "best_rmse": best_rmse,
            "best_r2": best_r2,
        }
