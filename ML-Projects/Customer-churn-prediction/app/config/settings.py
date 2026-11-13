from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODEL_DIR = BASE_DIR / "saved_models"

CHURN_DATASET_PATH = (
    RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)