from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "indian_banking_transactions.csv"

MODEL_DIR = BASE_DIR / "saved_models"
MODEL_PATH = MODEL_DIR / "fraud_detection_model.pkl"

TARGET_COLUMN = "is_fraud"

TEST_SIZE = 0.2
RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = None
N_JOBS = -1