from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "bank_marketing.csv"

MODEL_DIR = BASE_DIR / "saved_models"
MODEL_PATH = MODEL_DIR / "bank_marketing_prediction_model.pkl"

TARGET_COLUMN = "deposit"

TEST_SIZE = 0.2
RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = None
N_JOBS = -1