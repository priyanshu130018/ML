# 📞 Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn based on demographic, account, and service usage information. The project follows a modular architecture with separate components for data preprocessing, model training, evaluation, and prediction.

---

## 📌 Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning
- 🔄 Data Transformation
- 🤖 Model Training
- 📈 Model Evaluation
- 💾 Model Persistence using Joblib
- 🔮 Prediction using Saved Model
- 🏗 Modular Project Structure

---

## 📁 Project Structure

```
Customer-churn-prediction/
│
├── app/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── ml/
│   │   ├── data/
│   │   │   └── data_loader.py
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── data_cleaning.py
│   │   │   └── data_transformation.py
│   │   │
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   │   ├── evaluate.py
│   │   │   ├── model_provider.py
│   │   │   └── model_saver.py
│   │   │
│   │   ├── inference/
│   │   │   ├── model_loader.py
│   │   │   └── predict.py
│   │   │
│   │   └── pipelines/
│   │       └── training_pipeline.py
│   │
│   ├── utils/
│   └── main.py
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── saved_models/
│   └── churn_prediction_model.pkl
│
├── scripts/
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

This project uses the **Telco Customer Churn** dataset.

### Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

### Target

- **Churn**
  - Yes
  - No

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 🤖 Machine Learning Workflow

1. Load dataset
2. Clean missing values
3. Transform categorical and numerical features
4. Train classification model
5. Evaluate model performance
6. Save trained model
7. Predict customer churn using the saved model

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/priyanshu130018/Customer-churn-prediction.git
```

Navigate to the project

```bash
cd Customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python scripts/train.py
```

or

```bash
python -m app.main
```

---

## ▶️ Predict Customer Churn

Run the prediction module after training:

```bash
python -m app.main
```

---

## 📒 Exploratory Data Analysis

The project includes a comprehensive EDA notebook located at:

```
notebooks/eda.ipynb
```

### Analysis Performed

- Dataset overview
- Missing value analysis
- Target variable distribution
- Numerical feature analysis
- Correlation heatmap
- Categorical feature analysis
- Churn by Gender
- Churn by Contract Type
- Churn by Internet Service
- Churn by Payment Method
- Monthly Charges analysis
- Tenure analysis
- Pair plot visualization

### Key Insights

- The dataset contains **no missing values or duplicate records**.
- The target variable is **imbalanced**, with most customers not churning.
- Customers with **month-to-month contracts** show a higher churn rate.
- Customers paying higher monthly charges tend to churn more frequently.
- Longer-tenure customers are generally less likely to churn.
- Contract type and tenure are among the strongest indicators of customer churn.

---

## 📌 Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Feature selection
- Explainable AI (SHAP/LIME)
- REST API using FastAPI
- Docker containerization
- CI/CD pipeline
- Model deployment to cloud platforms

---

## 📄 License

This project is licensed under the MIT License.