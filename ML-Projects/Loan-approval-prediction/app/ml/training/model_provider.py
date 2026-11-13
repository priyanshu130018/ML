from sklearn.tree import DecisionTreeClassifier


class ModelProvider:
    
    def get_model(self):
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )
