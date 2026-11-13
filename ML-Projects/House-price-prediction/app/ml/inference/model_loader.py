from pathlib import Path

import joblib


class HouseModelLoader:
    def __init__(self, model_dir: Path | None = None):
        base_dir = Path(__file__).resolve().parents[3]
        self.model_dir = model_dir or (base_dir / "saved_models")

    def load(self):
        model_path = self.model_dir / "house_price_model.pkl"
        scaler_path = self.model_dir / "house_price_scaler.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        return model, scaler
