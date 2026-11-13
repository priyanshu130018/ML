from app.ml.pipelines.training_pipeline import HousePriceTrainingPipeline


class HousePriceTrainingRunner:
    
    def __init__(self):
        self.pipeline = HousePriceTrainingPipeline()

    def run(self):
        self.pipeline.run()


if __name__ == "__main__":
    HousePriceTrainingRunner().run()
