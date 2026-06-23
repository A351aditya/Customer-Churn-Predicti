from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('model_xgboost.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

app = FastAPI(
    title='Customer Churn Prediction API',
    description='Predicts telecom customer churn using XGBoost',
    version='1.0.0'
)

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    tenure_group: int
    charge_ratio: float
    num_services: int
    is_bundle: int
    senior_monthly: int

@app.get('/')
def home():
    return {'message': 'Churn Prediction API is running!'}

@app.post('/predict')
def predict(customer: CustomerData):
    # Convert to dataframe
    data = pd.DataFrame([customer.dict()])
    data = data[feature_names]
    
    # Scale
    data_scaled = scaler.transform(data)
    
    # Predict
    prob = model.predict_proba(data_scaled)[:, 1][0]
    prediction = int(prob >= 0.5)
    
    # Risk tier
    if prob >= 0.70:
        risk = 'High Risk'
    elif prob >= 0.40:
        risk = 'Medium Risk'
    else:
        risk = 'Low Risk'
    
    return {
        'churn_prediction': prediction,
        'churn_probability': round(float(prob), 4),
        'risk_tier': risk,
        'recommendation': 'Immediate retention action needed!' 
                         if risk == 'High Risk' else 
                         'Monitor closely' if risk == 'Medium Risk' 
                         else 'Low priority'
    }

@app.get('/health')
def health():
    return {'status': 'healthy'}

