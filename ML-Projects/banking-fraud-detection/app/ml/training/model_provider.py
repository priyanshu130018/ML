from sklearn.ensemble import RandomForestClassifier

from app.config.settings import (
    MAX_DEPTH,
    N_ESTIMATORS,
    N_JOBS,
    RANDOM_STATE,
)


class ModelProvider:

    def get_model(self):
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=N_JOBS,
        )