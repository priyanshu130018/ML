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
from app.ml.data.data_validation import DataValidation
from app.ml.preprocessing.feature_engineering import FeatureEngineer
from app.ml.preprocessing.data_transformation import DataTransformer
from app.ml.training.model_provider import ModelProvider
from app.ml.training.trainer import Trainer
from app.ml.training.evaluator import Evaluator
from app.ml.training.compare_models import CompareModels
from app.ml.training.model_saver import ModelSaver


class TrainingPipeline:

    def __init__(
        self,
        data_loader: DataLoader | None = None,
        data_cleaner: DataCleaner | None = None,
        validator: DataValidation | None = None,
        preprocessor: DataTransformer | None = None,
        model_provider: ModelProvider | None = None,
        trainer: Trainer | None = None,
        evaluator: Evaluator | None = None,
        compare_models: CompareModels | None = None,
        model_saver: ModelSaver | None = None,
    ):

        self.data_loader = data_loader or DataLoader(RAW_DATA_PATH)
        self.data_cleaner = data_cleaner or DataCleaner()
        self.validator = validator or DataValidation()
        self.preprocessor = preprocessor or DataTransformer()
        self.model_provider = model_provider or ModelProvider()
        self.trainer = trainer or Trainer()
        self.evaluator = evaluator or Evaluator()
        self.compare_models = compare_models or CompareModels()
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
    
        feature_engineer = FeatureEngineer()
        X_train_engineered = feature_engineer.transform(X_train.copy())
    
        preprocessor = self.preprocessor.build_pipeline(X_train_engineered)
    
        models = self.model_provider.get_models()
    
        pipelines = {}
        for name, model in models.items():
            pipelines[name] = Pipeline(
                steps=[
                    ("feature_engineer",feature_engineer,),
                    ("preprocessor",preprocessor,),
                    ("classifier",model,),
                ]
            )

        trained_models = self.trainer.train(
            pipelines,
            X_train,
            y_train,
        )

        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        evaluation_results = self.evaluator.evaluate(
            trained_models,
            X_test,
            y_test,
        )


        for model_name, pipeline in trained_models.items():

            classifier = pipeline.named_steps["classifier"]
            metrics = evaluation_results[model_name]

            print(f"Model          : {model_name}")
            print(f"Algorithm      : {classifier.__class__.__name__}")
            print(f"Features Used  : {classifier.n_features_in_}")
            print(f"Target Classes : {list(classifier.classes_)}")

            print("=" * 60)
            print(f"Accuracy       : {metrics['accuracy']:.4f}")
            print(f"Precision      : {metrics['precision']:.4f}")
            print(f"Recall         : {metrics['recall']:.4f}")
            print(f"F1 Score       : {metrics['f1_score']:.4f}")
            print(f"ROC AUC        : {metrics['roc_auc']:.4f}")
            print("-" * 60)

            print("Confusion Matrix")
            print(metrics["confusion_matrix"])
            print("-" * 60)

            print("Classification Report")
            print(metrics["classification_report"])

            print("=" * 60)


            

        best_model_name, best_model, best_metrics = self.compare_models.compare(
            trained_models,
            evaluation_results,
            metric="f1_score",
        )

        print()
        print("BEST MODEL SELECTED")
        print("=" * 60)
        print(f"Best Model Name : {best_model_name}")
        print(f"Algorithm  : {best_model.named_steps['classifier'].__class__.__name__}")
        print(f"Best F1    : {best_metrics['f1_score']:.4f}")

        self.model_saver.save(
            best_model,
            "bank_marketing_prediction_model",
        )

        print("=" * 60)
        print("Model saved successfully.")
        print("=" * 60)

        return (
            best_model_name,
            best_model,
            best_metrics,
            evaluation_results,
        )