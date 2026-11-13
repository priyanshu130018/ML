from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]

DATA_DIR = base_dir / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

DATA_FILE_PATH = RAW_DATA_DIR / "loan_data.csv"

MODEL_DIR = base_dir / "saved_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)