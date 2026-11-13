from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Evaluator:

    def evaluate(self, pipeline, X_test, y_test):
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "classification_report": classification_report(
                y_test,
                predictions,
            ),
        }