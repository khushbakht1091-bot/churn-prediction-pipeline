import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"


# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Fixtures 
@pytest.fixture(scope="module")
def model():
    model_uri = "models:/churn-logistic-regression/Production"
    return mlflow.sklearn.load_model(model_uri)

@pytest.fixture(scope="module")
def X_test():
    return pd.read_csv(DATA_DIR / "X_test.csv")


# ── Tests ──────────────────────────────────────────────────────────────────
def test_model_loads(model):
    assert model is not None

def test_model_predicts(model, X_test):
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)

def test_predictions_are_binary(model, X_test):
    predictions = model.predict(X_test)
    unique_values = set(np.unique(predictions))
    assert unique_values == {0, 1}

def test_predict_proba_shape(model, X_test):
    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)

def test_predict_proba_values_are_valid(model, X_test):
    probabilities = model.predict_proba(X_test)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0

