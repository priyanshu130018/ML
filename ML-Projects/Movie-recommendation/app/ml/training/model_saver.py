from pathlib import Path

import joblib

from app.config.settings import MODEL_DIR


class ModelSaver:

    def __init__(self, model_dir: Path | None = None,):
        self.model_dir = model_dir or MODEL_DIR
        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(self,model,model_name: str,):

        if not model_name.endswith(".pkl"):
            model_name += ".pkl"

        model_path = self.model_dir / model_name
        joblib.dump(model,model_path)

        return model_path