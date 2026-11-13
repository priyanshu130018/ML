from app.ml.data.data_loader import DataLoader
from app.ml.data.data_cleaner import DataCleaner
from app.ml.data.data_validator import DataValidator
from app.ml.preprocessing.matrix_builder import MatrixBuilder
from app.ml.preprocessing.sparse_converter import SparseConverter
from app.ml.training.model_provider import ModelProvider
from app.ml.training.trainer import Trainer
from app.ml.training.model_saver import ModelSaver


class TrainingPipeline:

    def __init__(self):

        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_validator = DataValidator()

        self.matrix_builder = MatrixBuilder()
        self.sparse_converter = SparseConverter()

        self.model_provider = ModelProvider()
        self.trainer = Trainer()
        self.model_saver = ModelSaver()

    def run(self):

        movies, ratings = self.data_loader.load_data()

        movies, ratings = self.data_cleaner.clean(
            movies,
            ratings,
        )

        movies, ratings = self.data_validator.validate(
            movies,
            ratings,
        )

        movie_user_matrix = self.matrix_builder.build(
            movies,
            ratings,
        )

        sparse_matrix = self.sparse_converter.convert(
            movie_user_matrix,
        )

        model = self.model_provider.get_model()

        trained_model = self.trainer.train(
            model,
            sparse_matrix,
        )

        print("=" * 60)
        print("Training Completed")
        print("=" * 60)

        self.model_saver.save(
            trained_model,
            "movie_recommendation_knn",
        )

        self.model_saver.save(
            movie_user_matrix,
            "movie_user_matrix",
        )

        
        print("Algorithm      : K-Nearest Neighbors (KNN)")
        print(f"Metric         : {trained_model.metric}")
        print(f"Neighbors (K)  : {trained_model.n_neighbors}")
        print(f"Movies         : {movie_user_matrix.shape[0]}")
        print(f"Users          : {movie_user_matrix.shape[1]}")
        print(f"Matrix Shape   : {movie_user_matrix.shape}")

        print("=" * 60)
        print("Model saved successfully!")
        print("=" * 60)

        return trained_model, movie_user_matrix