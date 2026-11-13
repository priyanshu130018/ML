class Trainer:

    def train(self, pipelines, X_train, y_train):

        trained_models = {}

        for model_name, pipeline in pipelines.items():

            pipeline.fit(X_train, y_train)

            trained_models[model_name] = pipeline

        return trained_models