from app.ml.inference.predict import HousePricePredictor
from app.utils.format_price import PriceFormatter


class HousePriceConsole:
    def __init__(self):
        self.predictor = HousePricePredictor()
        self.price_formatter = PriceFormatter()
        self.house = {
            "MedInc": 12.5,
            "HouseAge": 25,
            "AveRooms": 8.2,
            "AveBedrms": 1.1,
            "Population": 800,
            "AveOccup": 2.7,
            "Latitude": 37.75,
            "Longitude": -122.25,
        }

    def run(self):
        prediction = self.predictor.predict(self.house)

        print("=" * 50)
        print("HOUSE PRICE PREDICTION")
        print("=" * 50)
        print(f"Predicted Price : {self.price_formatter.format(prediction)}")
        print("=" * 50)


if __name__ == "__main__":
    HousePriceConsole().run()
