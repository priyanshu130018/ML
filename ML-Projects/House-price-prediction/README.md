# 🏠 House Price Prediction

An end-to-end Machine Learning project that predicts house prices based on property features. The project follows a modular architecture with separate components for data loading, model training, evaluation, model comparison, and inference.

---

## 📌 Features

* 📂 Modular project architecture
* 📊 Data loading and preprocessing
* 🤖 Multiple regression model comparison
* 📈 Model evaluation
* 💾 Model persistence using Joblib
* 🔮 Prediction using trained model
* 📝 Jupyter notebook for exploratory data analysis (EDA)

---

## 📁 Project Structure

```text
House-price-prediction/
│
├── app/
│   ├── ml/
│   │   ├── data/
│   │   │   └── data_loader.py
│   │   │
│   │   ├── inference/
│   │   │   ├── model_loader.py
│   │   │   └── predict.py
│   │   │
│   │   ├── pipelines/
│   │   │   └── training_pipeline.py
│   │   │
│   │   └── training/
│   │       ├── compare_model.py
│   │       ├── evaluate.py
│   │       ├── model_provider.py
│   │       ├── model_saver.py
│   │       └── trainer.py
│   │
│   ├── utils/
│   └── main.py
│
├── notebooks/
│   └── eda.ipynb
│
├── saved_models/
│    ├──house_price_model.pkl   
│    └──house_price_scaler.pkl
│
├── scripts/
│   └── train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

The project uses the **California Housing Dataset** available in **scikit-learn**.

Example features include:

* Median Income (MedInc)
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Average Occupancy
* Latitude
* Longitude

**Target**

* Median House Value (MedHouseVal)

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## 🤖 Machine Learning Workflow

1. Load dataset
2. Data preprocessing
3. Train multiple regression models
4. Compare model performance
5. Select the best model
6. Evaluate the final model
7. Save the trained model
8. Predict house prices using the saved model

---

## 📈 Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/priyanshu130018/House-price-prediction.git
```

Navigate into the project

```bash
cd House-price-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python -m app.main
```

---

## ▶️ Predict House Price

After training, the saved model can be used for inference through the prediction module.

```bash
python -m app.main
```

---

## 📒 Exploratory Data Analysis

The project includes a detailed exploratory data analysis notebook located at:

```text
notebooks/eda.ipynb
```

### Analysis Performed

* Dataset overview and structure
* Summary statistics
* Missing value analysis
* Distribution of numerical features
* Box plots for outlier detection
* Correlation heatmap
* Pair plot visualization
* Scatter plot analysis
* Target variable distribution
* Feature correlation with house prices

### Key Insights

* The dataset contains **no missing values**.
* Most numerical features have different value ranges.
* **Median Income (MedInc)** has the strongest positive correlation with house prices.
* Location-based features also show meaningful relationships with the target variable.
* Correlation analysis helped identify the most informative features for model training.

---

## 📌 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature engineering
* Flask/FastAPI deployment
* Docker support
* CI/CD pipeline

---

## 📄 License

This project is licensed under the MIT License.
