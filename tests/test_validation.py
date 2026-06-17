import pytest
import pandas as pd
from src.data.validate_data import validate_data

# Fixtures
@pytest.fixture
def good_data():
    return pd.DataFrame({
        "customerID":      ["001", "002", "003"],
        "tenure":          [12, 24, 6],
        "MonthlyCharges":  [50.0, 75.0, 30.0],
        "Contract":        ["Month-to-month", "One year", "Two year"],
        "Churn":           ["Yes", "No", "No"],
        "gender":          ["Male", "Female", "Male"],
        "InternetService": ["DSL", "Fiber optic", "No"],
    })

@pytest.fixture
def bad_data():
    return pd.DataFrame({
        "customerID":      ["001", "002", "003"],
        "tenure":          [-99, 24, 6],
        "MonthlyCharges":  [50.0, 75.0, 30.0],
        "Contract":        ["Month-to-month", "Weekly", "Two year"],
        "Churn":           ["Yes", "No", "No"],
        "gender":          ["Male", "Female", "Male"],
        "InternetService": ["DSL", "Fiber optic", "No"],
    })

# Tests
def test_validation_passes_on_good_data(good_data):
    try:
        validate_data(good_data)
    except ValueError:
        pytest.fail("validate_data raised ValueError on good data")

def test_validation_fails_on_bad_data(bad_data):
    with pytest.raises(ValueError):
        validate_data(bad_data)

def test_validation_error_message_is_informative(bad_data):
    with pytest.raises(ValueError) as exc_info:
        validate_data(bad_data)
    error_message = str(exc_info.value)
    assert "failed" in error_message.lower()