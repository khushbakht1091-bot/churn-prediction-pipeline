import sys
import os
import joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.models.model_registry import load_production_model

model = None
preprocessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    model = load_production_model()
    preprocessor = joblib.load("models/preprocessor.joblib")
    print("Model and preprocessor loaded successfully")
    yield
    print("Shutting down")



app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability",
    version="1.0.0",
    lifespan=lifespan
)

class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    SeniorCitizen: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: str

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok", "model": "churn-best-model", "stage": "Production"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_df = pd.DataFrame([customer.model_dump()])
    
    input_transformed = preprocessor.transform(input_df)
    
    probability = model.predict_proba(input_transformed)[0][1]
    
    prediction = "High Risk" if probability >= 0.5 else "Low Risk"
    
    return PredictionResponse(
        churn_probability=round(float(probability), 4),
        prediction=prediction
    )