class Recommender:

    def recommend(
        self,
        model,
        movie_user_matrix,
        movie_name,
        n_recommendations=5,
    ):

        movie_matches = movie_user_matrix.index[
            movie_user_matrix.index.str.contains(
                movie_name,
                case=False,
                na=False,
            )
        ]

        if movie_matches.empty:
            raise ValueError(
                f"No movie found matching '{movie_name}'."
            )

        movie_name = movie_matches[0]

        movie_index = movie_user_matrix.index.get_loc(
            movie_name
        )

        distances, indices = model.kneighbors(
            movie_user_matrix.iloc[[movie_index]],
            n_neighbors=n_recommendations + 1,
        )

        recommendations = []

        for index in indices.flatten()[1:]:
            recommendations.append(
                movie_user_matrix.index[index]
            )

        return recommendations