from app.ml.pipelines.recommendation_pipeline import (
    RecommendationPipeline,
)


class MovieRecommendationScript:

    def __init__(self):

        self.pipeline = RecommendationPipeline()

        self.movie_name = "Iron Man"

    def run(self):

        self.pipeline.recommend(
            self.movie_name,
        )


if __name__ == "__main__":
    MovieRecommendationScript().run()