from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.ml.data.data_loader import HousingDataLoader
from app.ml.training.model_provider import HousingModelProvider
from app.ml.training.trainer import HousingModelTrainer
from app.ml.training.compare_model import HousingModelComparer
from app.ml.training.model_saver import HouseModelSaver


class HousePriceTrainingPipeline:

    def __init__(
        self,
        data_loader: HousingDataLoader | None = None,
        comparer: HousingModelComparer | None = None,
        model_provider: HousingModelProvider | None = None,
        trainer: HousingModelTrainer | None = None,
        model_saver: HouseModelSaver | None = None,
        target_column: str = "MedHouseVal",
    ):
        self.data_loader = data_loader or HousingDataLoader()
        self.target_column = target_column
        self.comparer = comparer or HousingModelComparer()
        self.model_provider = model_provider or HousingModelProvider()
        self.trainer = trainer or HousingModelTrainer()
        self.model_saver = model_saver or HouseModelSaver()

    def run(self):

        df = self.data_loader.load()

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        models = self.model_provider.get_models()

        trained_models  = {}
        for name, model in models.items():
            trained_model = self.trainer.train(model, X_train_scaled, y_train)

            trained_models[name] = trained_model

        print("=" * 50)
        print("Training Completed")
        print("=" * 50)


        comparison = self.comparer.compare(trained_models, X_test_scaled, y_test)

        print("\nModel Comparison")
        print("-" * 75)
        print(f"{'Model':<20}{'MAE':>12}{'RMSE':>12}{'R2':>12}")
        print("-" * 75)

        for name, metrics in comparison["results"].items():
            print(
                f"{name:<20}"
                f"{metrics['MAE']:>12.4f}"
                f"{metrics['RMSE']:>12.4f}"
                f"{metrics['R2']:>12.4f}"
            )

        print("-" * 75)
        print("\nBest by Metric")
        print("-" * 40)
        print(f"Lowest MAE  : {comparison['best_mae']}")
        print(f"Lowest RMSE : {comparison['best_rmse']}")
        print(f"Highest R2  : {comparison['best_r2']}")
        print("-" * 40)

        print()
        print("=" * 50)
        print(f"Best Model : {comparison['best_name']}")
        print("=" * 50)

        self.model_saver.save(scaler, "house_price_scaler")
        self.model_saver.save(comparison["best_model"], "house_price_model")
        
        print("Model Saved Successfully")
        print("=" * 50)

        return (
            comparison["best_model"],
            comparison["best_name"],
            comparison["best_score"],
            comparison["results"],
            scaler
        )