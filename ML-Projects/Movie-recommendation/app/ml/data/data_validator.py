class DataValidator:

    MOVIE_COLUMNS = {
        "movieId",
        "title",
        "genres",
    }

    RATING_COLUMNS = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
    }

    def validate(
        self,
        movies,
        ratings,
    ):

        if not self.MOVIE_COLUMNS.issubset(movies.columns):
            missing = self.MOVIE_COLUMNS - set(movies.columns)
            raise ValueError(
                f"Missing movie columns: {missing}"
            )

        if not self.RATING_COLUMNS.issubset(ratings.columns):
            missing = self.RATING_COLUMNS - set(ratings.columns)
            raise ValueError(
                f"Missing rating columns: {missing}"
            )

        if movies.isnull().sum().sum() > 0:
            raise ValueError(
                "Movies dataset contains missing values."
            )

        if ratings.isnull().sum().sum() > 0:
            raise ValueError(
                "Ratings dataset contains missing values."
            )

        if movies["movieId"].duplicated().any():
            raise ValueError(
                "Duplicate movieId found."
            )

        if not ratings["rating"].between(
            0.5,
            5.0,
        ).all():
            raise ValueError(
                "Ratings must be between 0.5 and 5.0."
            )

        invalid_movies = (
            set(ratings["movieId"])
            - set(movies["movieId"])
        )

        if invalid_movies:
            raise ValueError(
                "Ratings contain invalid movieId values."
            )

        return movies, ratings