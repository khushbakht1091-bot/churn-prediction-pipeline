import sys
import os
import joblib
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from typing import Optional
from sklearn.pipeline import Pipeline

# Required for resolving src/ imports when running inside Docker,
# where the working directory may not be the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd
from src.models.model_registry import load_production_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

model: Optional[object] = None
preprocessor: Optional[Pipeline] = None

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence_score",
    "Distribution of churn probability scores returned by the model",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

CHURN_POSITIVE_PREDICTIONS = Counter(
    "churn_positive_predictions",
    "Total number of High Risk churn predictions made"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle.

    On startup: loads the Production model from MLflow registry and the
    preprocessor from disk. Both are stored as module-level globals so
    the predict endpoint can access them without reloading on every request.

    On shutdown: logs a shutdown message for observability.
    """
     
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
Instrumentator().instrument(app).expose(app)
                                         
class CustomerFeatures(BaseModel):
    """
    Pydantic model representing the input features for a single customer.

    Field names match the IBM Telco Customer Churn dataset exactly,
    as the preprocessor was trained on those column names. Renaming
    any field here will break the preprocessing pipeline.

    All categorical fields are validated against the exact values
    present in the training data to prevent silent prediction errors
    from unseen categories.
    """
    tenure: float = Field(ge=0, description="Months as customer, must be non-negative")
    MonthlyCharges: float = Field(gt=0, description="Monthly charge amount, must be positive")
    SeniorCitizen: int
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
    
    @field_validator('SeniorCitizen')
    @classmethod
    def validate_senior_citizen(cls, v):
        allowed = {0, 1}
        if v not in allowed:
             raise ValueError(f"Must be 0 or 1")
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
    """
    Returns the current health status of the API.

    Checks whether the model was successfully loaded during startup.
    Returns 200 if the service is ready to accept predictions.
    Returns 500 if the model failed to load during startup.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    # model name and stage are hardcoded here as a known simplification.
    # In production, these should be read dynamically from the loaded model metadata.
    return {"status": "ok", "model": "churn-best-model", "stage": "Production"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """
    Accepts customer features and returns a churn prediction.

    Transforms the input using the fitted preprocessor, then scores
    it with the Production model loaded from MLflow. Returns the
    churn probability and a risk label.

    Returns 500 if the model is not loaded or prediction fails.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        input_df = pd.DataFrame([customer.model_dump()])

        input_transformed = preprocessor.transform(input_df)

        probability = model.predict_proba(input_transformed)[0][1]

        prediction = "High Risk" if probability >= 0.5 else "Low Risk"
        
        PREDICTION_CONFIDENCE.observe(probability)

        if prediction == "High Risk":
            CHURN_POSITIVE_PREDICTIONS.inc()

        logger.info("Prediction: probability=%.4f, result=%s", probability, prediction)

        return PredictionResponse(
            churn_probability=round(float(probability), 4),
            prediction=prediction
        )
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed. See server logs.")