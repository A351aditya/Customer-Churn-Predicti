# Customer-Churn-Predicti
This project predicts whether a telecom customer will churn using historical data. It covers the full ML pipeline — from data cleaning and feature engineering to model training, explainability, business impact analysis, and API deployment.
#  Project Overview
This project predicts whether a telecom customer will churn using historical data. It covers the full ML pipeline — from data cleaning and feature engineering to model training, explainability, business impact analysis, and API deployment.
Dataset: Telco Customer Churn — Kaggle

# Problem Statement
Customer churn is a major challenge in the telecom industry. Retaining existing customers is significantly cheaper than acquiring new ones. This project builds a model that:
	•	Identifies high-risk customers before they churn
	•	Quantifies revenue at risk
	•	Provides actionable retention recommendations



# project structure
churn-prediction/
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.xls
├── customer_churn_prediction.ipynb   # Main notebook
├── app.py                            # FastAPI deployment
├── model_xgboost.pkl                 # Saved XGBoost model
├── scaler.pkl                        # Saved StandardScaler
├── feature_names.pkl                 # Feature names list
├── 01_churn_distribution.png
├── 02_eda_plots.png
├── 03_feature_engineering.png
├── 06_shap_summary.png
├── 07_shap_bar.png
├── 08_shap_waterfall.png
├── 09_shap_dependence.png
├── 10_business_impact.png
└── README.md

# pipeline steps 
|Step|Description                                |
|----|-------------------------------------------|
|1   |Load & Explore Data                        |
|2   |Exploratory Data Analysis (EDA)            |
|3   |Feature Engineering                        |
|4   |Preprocessing (Encoding, Scaling, SMOTE)   |
|5   |Model Training (LR, Random Forest, XGBoost)|
|6   |Evaluation (ROC-AUC, Precision, Recall, F1)|
|7   |SHAP Explainability                        |
|8   |Business Impact Analysis                   |
|9   |FastAPI Deployment                         |


# Feature Engineering
5 new features created to boost model performance:

|Feature         |Description                                          |
|----------------|-----------------------------------------------------|
|`tenure_group`  |Customer loyalty stage (New/Early/Mid/Loyal/Champion)|
|`charge_ratio`  |Monthly charges / Total charges                      |
|`num_services`  |Number of add-on services subscribed                 |
|`is_bundle`     |Has both internet + phone service                    |
|`senior_monthly`|High-risk segment: senior on month-to-month contract |

# model Trained
	•	Logistic Regression
	•	Random Forest (200 estimators)
	•	XGBoost ← Best Model

# Result
|Model              |Accuracy |Precision|Recall   |F1       |ROC-AUC  |
|-------------------|---------|---------|---------|---------|---------|
|Logistic Regression|~0.80    |~0.63    |~0.55    |~0.59    |~0.84    |
|Random Forest      |~0.82    |~0.67    |~0.58    |~0.62    |~0.86    |
|**XGBoost**        |**~0.83**|**~0.69**|**~0.60**|**~0.64**|**~0.87**|

# SHAP Explaninability

Used SHAP (SHapley Additive exPlanations) to explain model predictions:
	•	Global feature importance — which features matter most overall
	•	Waterfall plots — explain individual customer predictions
	•	Dependence plots — how tenure interacts with MonthlyCharges
Top churn drivers identified:
	1.	Contract type (month-to-month = highest risk)
	2.	Tenure (shorter = higher churn)
	3.	Monthly Charges (higher = higher churn)
	4.	Internet Service (Fiber optic = higher churn)
	5.	Payment Method (Electronic check = higher churn)

# Business Impact

|Metric                              |Value    |
|------------------------------------|---------|
|High Risk Customers (≥70%)          |~250     |
|Annual Revenue at Risk              |~$180,000|
|Retention Campaign Cost             |~$12,500 |
|Estimated Net Saving (30% retention)|~$41,500 |

# FastAPI Deployment
Run locally
pip install fastapi uvicorn joblib pydantic
uvicorn app:app --reload

# Teck Stack

	•	Python 3.11
	•	Pandas & NumPy — data manipulation
	•	Matplotlib & Seaborn — visualization
	•	Scikit-learn — preprocessing & evaluation
	•	imbalanced-learn — SMOTE
	•	XGBoost — best performing model
	•	SHAP — model explainability
	•	FastAPI — API deployment
	•	Joblib — model serialization

# Installation
# Clone the repo
git clone https://github.com/A351aditya/Customer-Churn-Prediction.git
cd churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook customer_churn_prediction.ipynb

# Or run the API
uvicorn app:app --reload

# Requirments
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
fastapi
uvicorn
joblib
pydantic

# Author
Aaditya panwar
 	•	GitHub:A351aditya


