class Trainer:

    def train(
        self,
        model,
        sparse_matrix,
    ):

        model.fit(sparse_matrix)

        return model