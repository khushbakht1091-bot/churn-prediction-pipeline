import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 10))

    with patch("src.api.main.load_production_model", return_value=mock_model), \
         patch("src.api.main.joblib.load", return_value=mock_preprocessor):
        with TestClient(app) as c:
            yield c


VALID_CUSTOMER = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
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
    "PaymentMethod": "Electronic check"
}


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input_returns_200(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_has_correct_fields(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    body = response.json()
    assert "churn_probability" in body
    assert "prediction" in body


def test_predict_probability_is_between_0_and_1(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    probability = response.json()["churn_probability"]
    assert 0.0 <= probability <= 1.0


def test_predict_prediction_label_is_valid(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    label = response.json()["prediction"]
    assert label in ("High Risk", "Low Risk")


def test_negative_tenure_returns_422(client):
    bad_input = {**VALID_CUSTOMER, "tenure": -1}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_zero_monthly_charges_returns_422(client):
    bad_input = {**VALID_CUSTOMER, "MonthlyCharges": 0}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_invalid_contract_returns_422(client):
    bad_input = {**VALID_CUSTOMER, "Contract": "banana"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_invalid_gender_returns_422(client):
    bad_input = {**VALID_CUSTOMER, "gender": "Unknown"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422


def test_missing_field_returns_422(client):
    bad_input = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422