from scipy.sparse import csr_matrix


class SparseConverter:

    def convert(
        self,
        movie_user_matrix,
    ):

        return csr_matrix(
            movie_user_matrix.values
        )