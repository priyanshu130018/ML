import pandas as pd


class MatrixBuilder:

    def build(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> pd.DataFrame:

        movie_data = pd.merge(
            ratings,
            movies,
            on="movieId",
        )

        movie_user_matrix = movie_data.pivot_table(
            index="title",
            columns="userId",
            values="rating",
            fill_value=0,
        )

        return movie_user_matrix