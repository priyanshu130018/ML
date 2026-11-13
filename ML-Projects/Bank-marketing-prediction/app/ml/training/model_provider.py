from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from app.config.settings import RANDOM_STATE


class ModelProvider:

    def get_models(self):
        return {
            "XGBoost": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "CatBoost": CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_state=RANDOM_STATE,
                verbose=False,
            ),
        }