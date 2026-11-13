# 💳 Loan Approval Prediction

An end-to-end Machine Learning project that predicts whether a loan application is approved based on an applicant's financial and personal information. The project follows a modular architecture with dedicated components for data validation, feature engineering, preprocessing, model training, evaluation, and inference.

---

## 📌 Features

- 📊 Exploratory Data Analysis (EDA)
- ✅ Data Validation
- ⚙️ Feature Engineering
- 🔄 Data Transformation
- 🤖 Model Training
- 📈 Model Evaluation
- 💾 Model Persistence using Joblib
- 🔮 Prediction Pipeline
- 🏗 Modular Project Architecture

---

## 📁 Project Structure

```
Loan-approval-prediction/
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
│   │   │   ├── predict.py
│   │   │   └── postprocessing.py
│   │   │
│   │   └── pipelines/
│   │       ├── training_pipeline.py
│   │       └── prediction_pipeline.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── loan_data.csv
│   │
│   └── processed/
│       └── eda_checked.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── saved_models/
│   └── loan_approval_model.pkl
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

The dataset contains applicant information used to predict loan approval.

### Features

- Person Age
- Gender
- Annual Income
- Employment Experience
- Home Ownership
- Loan Amount
- Loan Intent
- Loan Grade
- Loan Interest Rate
- Loan Percent Income
- Credit History Length
- Credit Score
- Previous Loan Defaults

### Engineered Features

- Income per Year of Experience
- Loan-to-Income Ratio
- Credit-to-Age Ratio
- Income-to-Age Ratio

### Target

- Loan Status

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
2. Validate data
3. Perform feature engineering
4. Transform features
5. Train Decision Tree classifier
6. Evaluate model performance
7. Save trained model
8. Load model for inference
9. Predict loan approval
10. Post-process prediction results

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/priyanshu130018/Loan-approval-prediction.git
```

Navigate into the project

```bash
cd Loan-approval-prediction
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

## ▶️ Predict Loan Approval

Run the prediction pipeline:

```bash
python -m app.main
```

---

## 📒 Exploratory Data Analysis

The project includes a detailed EDA notebook located at:

```
notebooks/eda.ipynb
```

### Analysis Performed

- Dataset overview
- Summary statistics
- Missing value analysis
- Duplicate record detection
- Invalid value checks
- Target variable distribution
- Numerical feature distributions
- Box plots for outlier detection
- Categorical feature analysis
- Correlation matrix
- Correlation heatmap
- Feature-wise analysis
- Clean dataset generation for model training

### Key Insights

- The dataset contains minimal data quality issues after validation.
- The target classes are moderately imbalanced, making evaluation metrics beyond accuracy important.
- Income, loan amount, interest rate, and credit score are among the most influential features.
- Correlation analysis revealed useful relationships for feature engineering.
- Engineered features such as loan-to-income ratio and income-to-age ratio improved the predictive information available to the model.
---

## 🧪 Testing

Test the trained model using:

```bash
python -m tests.test
```

---

## 📌 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Feature selection
- Model explainability using SHAP
- FastAPI/Flask deployment
- Docker support
- CI/CD pipeline
- Cloud deployment

---

## 📄 License

This project is licensed under the MIT License.