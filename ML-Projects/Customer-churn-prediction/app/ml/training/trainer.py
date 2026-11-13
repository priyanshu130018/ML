from sklearn.pipeline import Pipeline 

class ChurnModelTrainer:

    def train(self, pipeline: Pipeline, X_train, y_train) -> Pipeline:
        pipeline.fit(X_train, y_train)
        return pipeline