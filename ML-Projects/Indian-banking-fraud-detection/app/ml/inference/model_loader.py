from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

from app.config.settings import MODEL_DIR


class ModelLoader:

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or MODEL_DIR

    def load(self, model_name: str) -> Pipeline:
        if not model_name.endswith(".pkl"):
            model_name += ".pkl"

        model_path = self.model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}"
            )
        
        return joblib.load(model_path)