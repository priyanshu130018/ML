from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.data.data_loader import DataIngestion
from app.ml.data.data_validation import DataValidation
from app.ml.preprocessing.data_transformation import DataPreprocessor
from app.ml.preprocessing.feature_engineering import FeatureEngineer
from app.ml.training.model_provider import ModelProvider
from app.ml.training.trainer import ModelTrainer
from app.ml.training.evaluator import ModelEvaluator
from app.ml.training.model_saver import ModelSaver


class TrainingPipeline:

    def __init__(
        self,
        data_loader: DataIngestion | None = None,
        validator: DataValidation | None = None,
        preprocessor_builder: DataPreprocessor | None = None,
        model_provider: ModelProvider | None = None,
        trainer: ModelTrainer | None = None,
        evaluator: ModelEvaluator | None = None,
        model_saver: ModelSaver | None = None,
    ):
        self.data_loader = data_loader or DataIngestion()
        self.validator = validator or DataValidation()
        self.preprocessor_builder = (
            preprocessor_builder or DataPreprocessor()
        )
        self.model_provider = model_provider or ModelProvider()
        self.trainer = trainer or ModelTrainer()
        self.evaluator = evaluator or ModelEvaluator()
        self.model_saver = model_saver or ModelSaver()

    def run(self):
        df = self.data_loader.load_data()

        self.validator.validate(df)

        X = df.drop(columns=["loan_status"])
        y = df["loan_status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        preprocessor = self.preprocessor_builder.build_pipeline()
        model = self.model_provider.get_model()

        pipeline = Pipeline(
            steps=[
                ("feature_engineer", FeatureEngineer()),
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        trained_pipeline = self.trainer.train(
            pipeline,
            X_train,
            y_train,
        )

        print("=" * 60)
        print("Training Completed")
        print("=" * 60)

        classifier = trained_pipeline.named_steps["classifier"]

        print(f"Model       : {classifier.__class__.__name__}")
        print(f"Criterion   : {classifier.criterion}")
        print(f"Max depth   : {classifier.get_depth()}")
        print(f"Leaf nodes  : {classifier.get_n_leaves()}")
        print(f"Features    : {classifier.n_features_in_}")
        print(f"Classes     : {list(classifier.classes_)}")

        metrics = self.evaluator.evaluate(
            trained_pipeline,
            X_test,
            y_test,
        )

        print()
        print("-" * 60)
        print(f"Accuracy    : {metrics['accuracy']:.4f}")
        print(f"Precision   : {metrics['precision']:.4f}")
        print(f"Recall      : {metrics['recall']:.4f}")
        print(f"F1 Score    : {metrics['f1_score']:.4f}")
        print(f"ROC AUC     : {metrics['roc_auc']:.4f}")
        print("-" * 60)

        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print("-" * 60)

        print("Classification Report:")
        print(metrics["classification_report"])

        self.model_saver.save(
            trained_pipeline,
            "loan_approval_model",
        )

        print("=" * 60)
        print("Model Saved Successfully")
        print("=" * 60)

        return trained_pipeline, metrics