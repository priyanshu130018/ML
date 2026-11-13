from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

MOVIES_PATH = RAW_DATA_DIR / "movies.csv"
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"

MODEL_DIR = BASE_DIR / "saved_models"
MODEL_PATH = MODEL_DIR / "movie_recommendation_knn.pkl"

N_NEIGHBORS = 6
METRIC = "cosine"
ALGORITHM = "brute"