from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config.settings import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)

from app.ml.data.data_loader import DataLoader
from app.ml.data.data_cleaner import DataCleaner
from app.ml.data.data_validation import DataValidator
from app.ml.preprocessing.feature_engineering import FeatureEngineer
from app.ml.preprocessing.data_transformation import DataTransformer
from app.ml.training.model_provider import ModelProvider
from app.ml.training.trainer import Trainer
from app.ml.training.evaluator import Evaluator
from app.ml.training.model_saver import ModelSaver


class TrainingPipeline:

    def __init__(
        self,
        data_loader: DataLoader | None = None,
        data_cleaner: DataCleaner | None = None,
        validator: DataValidator | None = None,
        preprocessor: DataTransformer | None = None,
        model_provider: ModelProvider | None = None,
        trainer: Trainer | None = None,
        evaluator: Evaluator | None = None,
        model_saver: ModelSaver | None = None,
    ) -> None:

        self.data_loader = data_loader or DataLoader(RAW_DATA_PATH)
        self.data_cleaner = (data_cleaner or DataCleaner())
        self.validator = validator or DataValidator(TARGET_COLUMN)
        self.preprocessor = (preprocessor or DataTransformer())
        self.model_provider = (model_provider or ModelProvider())
        self.trainer = trainer or Trainer()
        self.evaluator = evaluator or Evaluator()
        self.model_saver = model_saver or ModelSaver()

    def run(self):

        df = self.data_loader.load()

        df = self.data_cleaner.clean(df)

        self.validator.validate(df)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        # Build preprocessing pipeline using engineered schema
        feature_engineer = FeatureEngineer()

        X_train_engineered = feature_engineer.transform( X_train.copy())

        preprocessor = self.preprocessor.build_pipeline(X_train_engineered)

        model = self.model_provider.get_model()

        pipeline = Pipeline(
            steps=[
                ("feature_engineer", feature_engineer,),
                ("preprocessor", preprocessor,),
                ("classifier", model,),
            ]
        )

        trained_pipeline = self.trainer.train(
            pipeline,
            X_train,
            y_train,
        )

        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        classifier = trained_pipeline.named_steps["classifier"]

        print(f"Model         : {classifier.__class__.__name__}")
        print(f"Trees         : {classifier.n_estimators}")
        print(f"Criterion     : {classifier.criterion}")
        print(f"Features      : {classifier.n_features_in_}")
        print(f"Classes       : {list(classifier.classes_)}")
        print(f"Max Features  : {classifier.max_features}")
        print(f"Bootstrap     : {classifier.bootstrap}")


        metrics = self.evaluator.evaluate(
            trained_pipeline,
            X_test,
            y_test,
        )

        print("-" * 60)
        print(f"Accuracy    : {metrics['accuracy']:.4f}")
        print(f"Precision   : {metrics['precision']:.4f}")
        print(f"Recall      : {metrics['recall']:.4f}")
        print(f"F1 Score    : {metrics['f1_score']:.4f}")
        print(f"ROC AUC     : {metrics['roc_auc']:.4f}")
        print("-" * 60)

        print("Confusion Matrix")
        print(metrics["confusion_matrix"])
        print("-" * 60)

        print("Classification Report")
        print(metrics["classification_report"])

        self.model_saver.save(
            trained_pipeline,
            "fraud_detection_model",
        )

        print("=" * 60)
        print("Model saved successfully.")
        print("=" * 60)

        return trained_pipeline, metrics