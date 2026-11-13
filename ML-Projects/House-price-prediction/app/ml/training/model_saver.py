from pathlib import Path

import joblib


class HouseModelSaver:

    def __init__(self, model_dir: Path | None = None):
        base_dir = Path(__file__).resolve().parents[3]
        self.model_dir = model_dir or (base_dir / "saved_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, model_name: str) -> Path:
        if not model_name.endswith(".pkl"):
            model_name += ".pkl"

        model_path = self.model_dir / model_name
        joblib.dump(model, model_path)
        
        return model_path
