from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline


class ChurnModelEvaluator:
    
    def evaluate(self, pipeline: Pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob[:, 1]),
        }
