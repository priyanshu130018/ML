from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class Evaluator:

    def evaluate(
        self,
        trained_models,
        X_test,
        y_test,
    ):

        results = {}

        for model_name, pipeline in trained_models.items():

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            results[model_name] = {
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "precision": precision_score(
                    y_test,
                    predictions,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                ),
                "f1_score": f1_score(
                    y_test,
                    predictions,
                ),
                "roc_auc": roc_auc_score(
                    y_test,
                    probabilities,
                ),
                "confusion_matrix": confusion_matrix(
                    y_test,
                    predictions,
                ),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                ),
            }

        return results