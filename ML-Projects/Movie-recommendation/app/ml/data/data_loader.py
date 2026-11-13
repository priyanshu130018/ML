import pandas as pd
from pathlib import Path

from app.config.settings import (
    MOVIES_PATH,
    RATINGS_PATH,
)


class DataLoader:

    def __init__(
        self,
        movies_path: Path | None = None,
        ratings_path: Path | None = None,
    ):
        self.movies_path = movies_path or MOVIES_PATH
        self.ratings_path = ratings_path or RATINGS_PATH

    def load_data(self):

        if not self.movies_path.exists():
            raise FileNotFoundError(
                f"Movies dataset not found at {self.movies_path}"
            )

        if not self.ratings_path.exists():
            raise FileNotFoundError(
                f"Ratings dataset not found at {self.ratings_path}"
            )

        movies = pd.read_csv(self.movies_path)
        ratings = pd.read_csv(self.ratings_path)

        return movies, ratings