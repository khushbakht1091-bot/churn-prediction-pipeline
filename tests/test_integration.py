import pytest
import psycopg2
import requests
import os

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://127.0.0.1:8000"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "churn_db",
    "user": "churn_user",
    "password": os.environ.get("CHURN_DB_PASSWORD"),
}


@pytest.mark.integration
def test_postgres_has_churn_data():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM raw_churn_data;")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    assert count > 0, f"Expected rows in customer_churn, got {count}"


@pytest.mark.integration
def test_mlflow_has_production_model():
    response = requests.get("http://127.0.0.1:5000/api/2.0/mlflow/registered-models/get?name=churn-best-model")
    assert response.status_code == 200
    body = response.json()
    assert "registered_model" in body

@pytest.mark.integration
def test_api_health_check():
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "churn-best-model"
    assert body["stage"] == "Production"

@pytest.mark.integration
def test_api_predict_valid_input():
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
        "NumAdminTickets": 0,
        "NumTechTickets": 1
    }
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "prediction" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in ("High Risk", "Low Risk")

@pytest.mark.integration
def test_api_predict_invalid_input():
    payload = {
        "gender": "Unknown",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
        "NumAdminTickets": 0,
        "NumTechTickets": 1
    }
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    assert response.status_code == 422