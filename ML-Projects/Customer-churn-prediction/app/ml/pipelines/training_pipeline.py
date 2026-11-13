from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.data.data_loader import ChurnDataLoader
from app.ml.data.data_cleaning import ChurnDataCleaner
from app.ml.preprocessing.data_transformation import ChurnPreprocessorBuilder
from app.ml.training.model_provider import ChurnModelProvider
from app.ml.training.evaluate import ChurnModelEvaluator
from app.ml.training.model_saver import ChurnModelSaver
from app.ml.training.trainer import ChurnModelTrainer


class ChurnTrainingPipeline:

    def __init__(
        self,
        data_loader: ChurnDataLoader | None = None,
        cleaner: ChurnDataCleaner | None = None,
        preprocessor_builder: ChurnPreprocessorBuilder | None = None,
        model_provider: ChurnModelProvider | None = None,
        trainer: ChurnModelTrainer | None = None,
        evaluator: ChurnModelEvaluator | None = None,
        model_saver: ChurnModelSaver | None = None,
    ):
        self.data_loader = data_loader or ChurnDataLoader()
        self.cleaner = cleaner or ChurnDataCleaner()
        self.preprocessor_builder = preprocessor_builder or ChurnPreprocessorBuilder()
        self.model_provider = model_provider or ChurnModelProvider()
        self.trainer = trainer or ChurnModelTrainer()
        self.evaluator = evaluator or ChurnModelEvaluator()
        self.model_saver = model_saver or ChurnModelSaver()

    def run(self):
        df = self.data_loader.load()
        df = self.cleaner.clean(df)

        X = df.drop(columns="Churn")
        y = df["Churn"].map({"No": 0, "Yes": 1})

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        preprocessor = self.preprocessor_builder.build_pipeline()

        model = self.model_provider.get_model()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )
        trained_pipeline = self.trainer.train(pipeline, X_train, y_train)
        print("=" * 60)
        print("Training Completed")
        print("=" * 60)

        classifier = trained_pipeline.named_steps["classifier"]

        print(f"Model       : {classifier.__class__.__name__}")
        print(f"Features    : {classifier.n_features_in_}")
        print(f"Classes     : {list(classifier.classes_)}")
        print(f"Solver        : {classifier.solver}")
        print(f"Penalty       : {classifier.penalty}")
        print(f"Intercept   : {classifier.intercept_}")
        print(f"Coefficients: {classifier.coef_.shape}")

        
        metrics = self.evaluator.evaluate(trained_pipeline, X_test, y_test)

        print()
        print("-" * 60)
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print("-" * 60)

        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print("-" * 60)

        print("Classification Report:")
        print(metrics["classification_report"])

        self.model_saver.save(trained_pipeline, "churn_prediction_model")
        print("=" * 60)
        print("Model Saved Successfully")
        print("=" * 60)

        return trained_pipeline, metrics
