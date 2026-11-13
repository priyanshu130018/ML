# Machine Learning Projects

A collection of machine learning notebooks and end-to-end projects covering fundamental algorithms, advanced models, practical workflows, and real-world applications.

## Contents

- Supervised Learning
  - Classification
  - Regression
- Unsupervised Learning
  - Clustering
  - Dimensionality Reduction
- Semi-Supervised Learning
- Anomaly Detection
- Recommendation Systems
- Reinforcement Learning
- Time Series Forecasting
- End-to-End ML Projects

## Repository Structure

```text
ML/
├── Anomaly_Detection/
├── ML-Projects/
├── Recommendation_Systems/
├── Reinforcement_Learning/
├── Semi-Supervised_Learning/
├── Supervised_Learning/
│   ├── Classification/
│   └── Regression/
├── Time_Series/
├── Unsupervised_Learning/
│   ├── Clustering/
│   └── Dimensionality Reduction/
├── .gitignore
└── README.md
```

## End-to-End Projects

The [`ML-Projects`](./ML-Projects) directory contains modular projects with:

- Data loading and validation
- Data cleaning and preprocessing
- Feature engineering
- Model training and comparison
- Model evaluation
- Model persistence
- Prediction pipelines
- Exploratory data analysis
- Unit tests

Projects include:

- Bank Marketing Prediction
- Customer Churn Prediction
- Email Spam Detection
- House Price Prediction
- Indian Banking Fraud Detection
- Loan Approval Prediction
- Movie Recommendation
- Banking Fraud Detection
- Medical Insurance Cost Prediction

Each project may include its own `README.md` and `requirements.txt`.

## Machine Learning Topics

### Supervised Learning

- Linear Regression
- Ridge and Lasso Regression
- Elastic Net
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Support Vector Machines
- K-Nearest Neighbors
- Naive Bayes

### Unsupervised Learning

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis
- Dimensionality Reduction

### Other Topics

- Anomaly Detection
- Recommendation Systems
- Reinforcement Learning
- Semi-Supervised Learning
- Time Series Forecasting
- Deep Learning workflows

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Joblib
- Jupyter Notebook

## Getting Started

Clone the repository:

```bash
git clone https://github.com/priyanshu130018/ML.git
cd ML
```

Create a virtual environment:

```bash
python -m venv .MLvenv
```

Activate it on Windows:

```bash
.MLvenv\Scripts\activate
```

Install dependencies for an individual project:

```bash
pip install -r requirements.txt
```

Run a training script:

```bash
python scripts/train.py
```

Open the notebooks:

```bash
jupyter notebook
```

## Data and Model Files

Datasets, trained models, logs, virtual environments, and generated outputs are excluded through `.gitignore`.

For large datasets or trained models, use a suitable storage service instead of committing them directly to GitHub.

## License

This repository is intended for educational and portfolio purposes. Add a specific license before publishing if required.

## Dataset Policy

Datasets must be downloaded separately and stored in local data/raw/ or data/processed/ folders. Dataset files, trained models, logs, virtual environments, and generated outputs are excluded by .gitignore and must not be uploaded to GitHub.
