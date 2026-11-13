Stage 1: Regression

Linear → Ridge → Lasso → Elastic Net → SVR

Stage 2: Classical Trees

Decision Tree → Random Forest → Extra Trees → Gradient Boosting

Stage 3: Advanced Trees

XGBoost → LightGBM → CatBoost → Tuning → Explainability

Stage 4: Classification

Logistic Regression → KNN → Naive Bayes → SVM → Trees → Ensembles

Stage 5: Unsupervised Learning

K-Means → Hierarchical Clustering → DBSCAN → PCA

Stage 6: Deep Learning

MLP → CNN → RNN/LSTM → Transformers

Stage 7: GenAI

Embeddings → Vector DB → RAG → LLM → PDF Chatbot





s.no	  Model	             |    Project	            |   Dataset
1	Linear Regression	     | House Price Prediction	| California Housing Dataset
2	Logistic Regression	     | Customer Churn	        | IBM Telco Customer Churn Dataset
3	Decision Tree	         | Loan Approval	        | Loan Prediction Dataset (Kaggle)
4	Random Forest	         | Credit Risk	            | German Credit Dataset
5	XGBoost	                 | Sales Prediction         | Rossmann Store Sales
6	KNN                      | Movie Recommendation     | MovieLens
7	SVM		                 | Spam Detection           | SMS Spam Collection
8	K-Means	                 | Customer Segmentation	| Mall Customers Dataset
9	PCA	                     | Dimensionality Reduction	| Wine Dataset
10	Neural Network	         | Digit Recognition	    | MNIST
11	CNN	                     | Image Classification	    | CIFAR-10
12	Transformer	             | Text Classification	    | AG News
13	LLM/RAG	PDF              | Chatbot	                | Your own PDF documents
14	Time Series              | Forecasting	            | Store Sales Time Series
15	Recommendation System	 | Movie Recommendation	    | MovieLens
## Dataset Policy

Datasets are downloaded locally into each project's data/raw/ directory and are ignored by Git. Do not commit CSV, TSV, Excel, parquet, or other dataset files. Trained models and experiment logs are also ignored.

## Running a Project

`ash
cd <project-folder>
pip install -r requirements.txt
python scripts/train.py
`

See each project README for its purpose, dataset source, local data path, and test command.
