from app.ml.pipelines.training_pipeline import TrainingPipeline


class LoanTrainingScript:
    def __init__(self):
        self.pipeline = TrainingPipeline()

    def run(self):
        self.pipeline.run()


if __name__ == "__main__":
    LoanTrainingScript().run()
    
