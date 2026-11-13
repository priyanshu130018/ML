from pathlib import Path
import joblib

from app.config.settings import MODEL_DIR


class ModelSaver:

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(self, pipeline, model_name: str) -> Path:
        if not model_name.endswith(".pkl"):
            model_name += ".pkl"

        model_path = self.model_dir / model_name
        joblib.dump(pipeline, model_path)

        return model_path