class CompareModels:

    def compare(
        self,
        trained_models: dict,
        evaluation_results: dict,
        metric: str = "f1_score",
    ):
        best_model_name = max(
            evaluation_results,
            key=lambda model: evaluation_results[model][metric],
        )

        best_model = trained_models[best_model_name]

        best_metrics = evaluation_results[best_model_name]

        return (
            best_model_name,
            best_model,
            best_metrics,
        )