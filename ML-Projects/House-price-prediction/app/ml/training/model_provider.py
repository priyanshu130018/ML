from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class HousingModelProvider:

    def get_models(self):
        return {
            "Linear Regression": LinearRegression(),

            "Ridge Regression": Ridge(alpha=10),

            "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),

            "Decision Tree": DecisionTreeRegressor(
                max_depth=5,
                random_state=42
            ),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
        }