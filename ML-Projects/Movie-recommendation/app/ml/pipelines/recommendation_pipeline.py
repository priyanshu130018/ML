from app.ml.inference.model_loader import ModelLoader
from app.ml.inference.recommender import Recommender
from app.ml.inference.postprocessor import PostProcessor


class RecommendationPipeline:

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        recommender: Recommender | None = None,
        postprocessor: PostProcessor | None = None,
    ):

        self.model_loader = model_loader or ModelLoader()

        self.model = self.model_loader.load(
            "movie_recommendation_knn"
        )

        self.movie_user_matrix = self.model_loader.load(
            "movie_user_matrix"
        )

        self.recommender = recommender or Recommender()
        self.postprocessor = postprocessor or PostProcessor()

    def recommend(
        self,
        movie_name: str,
    ):

        recommendations = self.recommender.recommend(
            self.model,
            self.movie_user_matrix,
            movie_name,
        )

        result = self.postprocessor.process(
            movie_name,
            recommendations,
        )

        print("=" * 60)
        print("MOVIE RECOMMENDATION SYSTEM")
        print("=" * 60)
        print(f"Movie: {result['movie']}")
        print()

        for i, movie in enumerate(
            result["recommendations"],
            start=1,
        ):
            print(f"{i}. {movie}")

        print("=" * 60)

        return result