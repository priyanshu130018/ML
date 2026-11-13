from sklearn.neighbors import NearestNeighbors

from app.config.settings import (
    ALGORITHM,
    METRIC,
    N_NEIGHBORS,
)


class ModelProvider:

    def get_model(self):

        return NearestNeighbors(
            n_neighbors=N_NEIGHBORS,
            metric=METRIC,
            algorithm=ALGORITHM,
        )