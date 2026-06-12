import sys
import os
import joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd
from src.models.model_registry import load_production_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = None
preprocessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    model = load_production_model()
    preprocessor = joblib.load("models/preprocessor.joblib")
    logger.info("Model and preprocessor loaded successfully")
    yield
    logger.info("Shutting down")



app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability",
    version="1.0.0",
    lifespan=lifespan
)

class CustomerFeatures(BaseModel):
    tenure: float = Field(ge=0, description="Months as customer, must be non-negative")
    MonthlyCharges: float = Field(gt=0, description="Monthly charge amount, must be positive")
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

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        allowed = {'Male', 'Female'}
        if v not in allowed:
            raise ValueError(f"Must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    @classmethod
    def validate_yes_no(cls, v):
        allowed = {'Yes', 'No'}
        if v not in allowed:
            raise ValueError(f"Must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator('MultipleLines')
    @classmethod
    def validate_multiple_lines(cls, v):
        allowed = {'Yes', 'No', 'No phone service'}
        if v not in allowed:
            raise ValueError(f"Must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator('InternetService')
    @classmethod
    def validate_internet_service(cls, v):
        allowed = {'DSL', 'Fiber optic', 'No'}
        if v not in allowed:
            raise ValueError(f"Must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator('OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies')
    @classmethod
    def validate_internet_features(cls, v):
        allowed = {'Yes', 'No', 'No internet service'}
        if v not in allowed:
            raise ValueError(f"Must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator('Contract')
    @classmethod
    def validate_contract(cls, v):
        allowed = {'Month-to-month', 'One year', 'Two year'}
        if v not in allowed:
            raise ValueError(f"Must be one of: {', '.join(sorted(allowed))}")
        return v

    @field_validator('PaymentMethod')
    @classmethod
    def validate_payment_method(cls, v):
        allowed = {'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'}
        if v not in allowed:
            raise ValueError(f"Must be one of: {', '.join(sorted(allowed))}")
        return v

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
    
    logger.info("Prediction: probability=%.4f, result=%s", probability, prediction)

    return PredictionResponse(
        churn_probability=round(float(probability), 4),
        prediction=prediction
    )