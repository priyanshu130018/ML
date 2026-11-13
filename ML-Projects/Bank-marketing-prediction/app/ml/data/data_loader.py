import pandas as pd
from pathlib import Path

from app.config.settings import RAW_DATA_PATH


class DataLoader:

    def __init__(self, file_path: Path | None = None):
        self.file_path = file_path or RAW_DATA_PATH

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        return pd.read_csv(self.file_path)