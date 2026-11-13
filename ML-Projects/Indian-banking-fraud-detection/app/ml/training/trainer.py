class Trainer:

    def train(self, pipeline, X_train, y_train):
        pipeline.fit(X_train, y_train)
        return pipeline