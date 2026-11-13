
from app.ml.pipelines.training_pipeline import ChurnTrainingPipeline


class ChurnTrainingScript:
    def __init__(self):
        self.pipeline = ChurnTrainingPipeline()

    def run(self):
        self.pipeline.run()


if __name__ == "__main__":
    ChurnTrainingScript().run()
