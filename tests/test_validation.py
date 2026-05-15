import pytest
import pandas as pd
from pathlib import Path
from src.data.validate_data import validate_data

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "telco_churn_cleaned.csv"

# Fixtures
@pytest.fixture
def good_data():
    return pd.read_csv(DATA_PATH)

@pytest.fixture
def bad_data():
    df = pd.read_csv(DATA_PATH).copy()
    df.loc[0, "tenure"] = -99
    df.loc[1, "Contract"] = "Weekly"
    return df


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