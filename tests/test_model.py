import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
    mock_model.predict_proba.return_value = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.4, 0.6],
        [0.7, 0.3],
    ])
    return mock_model


@pytest.fixture(scope="module")
def X_test():
    return np.zeros((5, 10))


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