class DataCleaner:

    def clean(
        self,
        movies,
        ratings,
    ):

        movies = movies.drop_duplicates()
        ratings = ratings.drop_duplicates()

        movies.columns = movies.columns.str.strip()
        ratings.columns = ratings.columns.str.strip()

        object_columns = movies.select_dtypes(
            include=["object"]
        ).columns

        for column in object_columns:
            movies[column] = movies[column].str.strip()

        return movies, ratings