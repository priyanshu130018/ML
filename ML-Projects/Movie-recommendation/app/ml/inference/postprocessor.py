class PostProcessor:

    def process(
        self,
        movie_name: str,
        recommendations: list,
    ):

        return {
            "movie": movie_name,
            "recommendations": recommendations,
        }