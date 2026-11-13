# 📈 Bank Marketing Prediction

An end-to-end Machine Learning project that predicts whether a customer will subscribe to a term deposit based on demographic, financial, and previous marketing campaign information. The project follows a modular architecture with dedicated components for data loading, validation, preprocessing, feature engineering, model training, evaluation, model comparison, and inference.

---

## 📌 Features

* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data Cleaning
* ✅ Data Validation
* ⚙️ Feature Engineering
* 🔄 Data Transformation
* 🤖 Multiple Model Training
* 📈 Model Evaluation
* 🏆 Best Model Selection
* 💾 Model Persistence using Joblib
* 🔮 Prediction Pipeline
* 🏗 Modular Project Architecture

---

## 📁 Project Structure

```text
Bank-marketing-prediction/
│
├── app/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── ml/
│   │   ├── data/
│   │   │   ├── data_loader.py
│   │   │   ├── data_cleaner.py
│   │   │   └── data_validator.py
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── feature_engineer.py
│   │   │   └── data_transformer.py
│   │   │
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
│   │   │   ├── compare_models.py
│   │   │   ├── model_provider.py
│   │   │   └── model_saver.py
│   │   │
│   │   ├── inference/
│   │   │   ├── model_loader.py
│   │   │   ├── predictor.py
│   │   │   └── postprocessor.py
│   │   │
│   │   └── pipelines/
│   │       ├── training_pipeline.py
│   │       └── prediction_pipeline.py
│   │
│   └── main.py
│
├── data/
│   └── raw/
│       └── bank_marketing.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── saved_models/
│   └── bank_marketing_prediction_model.pkl
│
├── scripts/
│   └── train.py
│
├── tests/
│   └── test.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset contains customer information collected during direct marketing campaigns conducted by a banking institution.

### Features

* Age
* Job
* Marital Status
* Education
* Credit Default
* Account Balance
* Housing Loan
* Personal Loan
* Contact Type
* Contact Day
* Contact Month
* Call Duration
* Campaign Contacts
* Days Since Previous Contact
* Previous Contacts
* Previous Campaign Outcome

### Engineered Features

* Balance per Age
* Campaign Efficiency
* Contact Frequency
* Previous Contact Ratio

### Target

* Deposit Subscription

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Joblib
* Jupyter Notebook

---

## 🤖 Machine Learning Workflow

1. Load dataset
2. Clean data
3. Validate data
4. Split training and testing data
5. Perform feature engineering
6. Transform features
7. Train multiple machine learning models
8. Evaluate model performance
9. Compare models using F1-Score
10. Save the best-performing model
11. Load model for inference
12. Predict customer deposit subscription
13. Post-process prediction results

---

## 📈 Models Used

* XGBoost Classifier
* LightGBM Classifier
* CatBoost Classifier

The best model is automatically selected based on the highest **F1-Score**.

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/priyanshu130018/Bank-marketing-prediction.git
```

Navigate into the project

```bash
cd Bank-marketing-prediction
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

## ▶️ Predict Customer Subscription

```bash
python -m app.main
```

---

## 📒 Exploratory Data Analysis

The project includes a detailed EDA notebook located at:

```text
notebooks/eda.ipynb
```

### Analysis Performed

* Dataset overview
* Summary statistics
* Missing value analysis
* Duplicate record detection
* Data cleaning
* Target distribution
* Numerical feature analysis
* Categorical feature analysis
* Correlation analysis
* Feature engineering insights
* Clean dataset generation

### Key Insights

* Customer contact duration strongly influences deposit subscription.
* Previous campaign outcomes provide valuable predictive information.
* Account balance and customer demographics contribute to prediction performance.
* Feature engineering improves model effectiveness.
* Multiple gradient boosting models achieve strong predictive performance.

---

## 🧪 Testing

Test the trained model using:

```bash
python -m tests.test
```

---

## 📌 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature selection
* SHAP explainability
* FastAPI deployment
* Docker support
* CI/CD pipeline
* Cloud deployment

---

## 📄 License

This project is licensed under the MIT License.

## Dataset and Usage

- Dataset: [Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- Local path: data/raw/bank_marketing.csv (download separately; datasets are not committed)

`ash
pip install -r requirements.txt
python scripts/train.py
python -m tests.test
`
"@
    'Customer-churn-prediction' = @"

## Dataset and Usage

- Dataset: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Local path: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv (download separately)

`ash
pip install -r requirements.txt
python scripts/train.py
`
"@
    'Email-spam-detection' = @"

## Dataset and Usage

- Dataset: [SMS Spam Collection on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Local path: data/raw/spam.csv (download separately)

`ash
pip install -r requirements.txt
python scripts/train.py
python -m tests.test
`
"@
    'House-price-prediction' = @"

## Dataset and Usage

- Dataset: [House Prices on Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Local data path depends on the notebook/application configuration; download the dataset separately.

`ash
pip install -r requirements.txt
python scripts/train.py
`
"@
    'Indian-banking-fraud-detection' = @"

## Dataset and Usage

- Dataset: [Indian Banking Dataset search on Kaggle](https://www.kaggle.com/search?q=indian+banking+transactions+dataset)
- Local path: data/raw/indian_banking_transactions.csv (download separately)

`ash
pip install -r requirements.txt
python scripts/train.py
python -m tests.Permutation_importance_test
`
"@
    'Loan-approval-prediction' = @"

## Dataset and Usage

- Dataset: [Loan Approval Classification Dataset on Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- Local path: data/raw/loan_data.csv (download separately)

`ash
pip install -r requirements.txt
python scripts/train.py
python -m tests.test
`
"@
    'Movie-recommendation' = @"

## Dataset and Usage

- Dataset: [MovieLens Latest Small on GroupLens](https://grouplens.org/datasets/movielens/)
- Kaggle mirror: [MovieLens Dataset on Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)
- Local paths: data/raw/movies.csv and data/raw/ratings.csv (download separately)

`ash
pip install -r requirements.txt
python scripts/train.py
python -m tests.test
`
"@
    'banking-fraud-detection' = @"

## Dataset and Usage

- Dataset: [German Credit Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/german-credit)
- Banking transactions: [Kaggle dataset search](https://www.kaggle.com/search?q=indian+banking+transactions+dataset)
- Local paths: data/raw/german_credit_data.csv and data/raw/indian_banking_transactions.csv

`ash
pip install -r requirements.txt
python scripts/train.py
python -m tests.Permutation_importance_test
`
"@
    'Medical-Insurance-Cost-Prediction' = @"

## Dataset and Usage

- Dataset: [Medical Cost Personal Dataset on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Local path: data/insurance.csv (download separately; do not commit it)

Open 	rain.ipynb after downloading the dataset into the data directory.
