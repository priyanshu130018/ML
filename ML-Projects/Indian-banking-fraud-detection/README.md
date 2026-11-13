# 🏦 Indian Banking Fraud Detection

An end-to-end Machine Learning project that predicts whether a banking transaction is fraudulent using a **Random Forest Classifier**. The project follows a production-style architecture with data validation, preprocessing, feature engineering, model training, evaluation, and an inference pipeline.

---

## 📌 Project Overview

Financial fraud causes significant losses for banks and customers. This project builds a fraud detection system that analyzes transaction details and predicts whether a transaction is **Fraudulent** or **Legitimate**.

The project demonstrates the complete machine learning workflow, from raw data preprocessing to deploying a prediction pipeline.

---

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Data Validation
- Missing Value Handling
- Feature Engineering
- Data Preprocessing
- Random Forest Classification
- Model Evaluation
- Feature Importance Analysis
- Model Serialization
- Prediction Pipeline
- Probability-Based Predictions
- Modular Production-Style Codebase

---

## 📁 Project Structure

```text
Indian-Banking-Fraud-Detection/
│
├── app/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── ml/
│   │   ├── data/
│   │   │   ├── data_loader.py
│   │   │   └── data_validation.py
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── data_transformation.py
│   │   │   └── feature_engineering.py
│   │   │
│   │   ├── training/
│   │   │   ├── trainer.py
│   │   │   ├── evaluator.py
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
│   ├── utils/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── saved_models/
│
├── scripts/
│   └── train.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project uses an **Indian Banking Transactions Dataset** containing transaction details and a fraud label.

Typical features include:

- Transaction ID
- Customer ID
- Transaction Amount
- Transaction Type
- Payment Method
- Merchant Category
- Transaction Date
- Account Information
- Device Details
- Location Information

**Target Variable**

- Fraudulent Transaction
- Legitimate Transaction

---

## 🔍 Exploratory Data Analysis

The project includes a complete EDA notebook located at:

```text
notebooks/eda.ipynb
```

### Analysis Performed

- Dataset Overview
- Summary Statistics
- Missing Value Analysis
- Duplicate Detection
- Data Type Inspection
- Unique Value Analysis
- Target Distribution
- Numerical Feature Analysis
- Categorical Feature Analysis
- Numerical Features vs Target
- Categorical Features vs Target
- Correlation Heatmap
- Outlier Detection
- Skewness Analysis
- Feature Engineering Exploration
- Data Validation
- Clean Dataset Generation

### Key Insights

- Checked data quality before model training.
- Identified missing values and duplicate records.
- Explored fraud class distribution.
- Detected outliers in numerical variables.
- Examined relationships between transaction features and fraud.
- Generated insights for feature engineering.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

- Missing Value Handling
- Duplicate Removal
- Feature Engineering
- Encoding Categorical Features
- Scaling Numerical Features
- Train-Test Split

---

## 🧠 Model

The project uses:

**Random Forest Classifier**

Reasons for choosing Random Forest:

- Handles nonlinear relationships
- Robust to noise
- Reduces overfitting
- Works well with mixed feature types
- Provides feature importance scores

---

## 🔄 Project Workflow

```text
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Validation
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Train-Test Split
      │
      ▼
Random Forest Model
      │
      ▼
Model Evaluation
      │
      ▼
Save Model
      │
      ▼
Prediction Pipeline
```

---

## 📈 Evaluation Metrics

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Indian-Banking-Fraud-Detection.git

cd Indian-Banking-Fraud-Detection
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python scripts/train.py
```

The trained model will be saved inside:

```text
saved_models/
```

---

## 🔮 Prediction

Run:

```bash
python app/main.py
```

Example output:

```text
============================================================
INDIAN BANKING FRAUD DETECTION
============================================================

Prediction : Legitimate Transaction

Confidence : 98.74%

------------------------------------------------------------
Probabilities
------------------------------------------------------------

Fraud        : 1.26%
Legitimate   : 98.74%
```

---

## 📊 Feature Engineering

Examples of engineered features:

- Transaction Frequency
- High Value Transaction Flag
- Amount-Based Features
- Time-Based Features
- Customer-Level Transaction Statistics

---

## 🧪 Testing

The project includes testing utilities for:

- Data Validation
- Feature Engineering
- Model Prediction
- Pipeline Verification

---

## 🔮 Future Improvements

- Hyperparameter Tuning
- XGBoost / LightGBM Comparison
- SHAP Explainability
- Cross Validation
- Class Imbalance Handling (SMOTE)
- Model Monitoring
- Streamlit Web Application
- Docker Deployment

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📜 License

This project is intended for educational and portfolio purposes.