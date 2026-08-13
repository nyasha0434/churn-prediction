from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0"
)

model = joblib.load("saved_models/xgboost_churn_model.pkl")
with open("saved_models/feature_names.json") as f:
    feature_names = json.load(f)

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    PaperlessBilling: str
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaymentMethod: str

@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])

    df_input['gender'] = df_input['gender'].map({'Male': 1, 'Female': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})

    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    df_input = pd.get_dummies(df_input, columns=multi_cols)

    df_input = df_input.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(float(probability), 4)
    }