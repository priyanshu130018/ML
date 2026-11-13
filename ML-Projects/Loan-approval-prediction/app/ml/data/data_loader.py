import pandas as pd
from pathlib import Path

from app.config.settings import DATA_FILE_PATH


class DataIngestion:

    def __init__(self, file_path: Path | None = None):
       self.file_path = file_path or DATA_FILE_PATH

    def load_data(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.file_path}"
            )

        return pd.read_csv(self.file_path)