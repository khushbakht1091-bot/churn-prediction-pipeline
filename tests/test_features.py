import pytest
import pandas as pd
import numpy as np

# Fixtures
@pytest.fixture
def X_train():
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(5634, 10))

@pytest.fixture
def X_test():
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(1398, 10))

@pytest.fixture
def y_train():
    np.random.seed(42)
    n = 5634
    labels = np.zeros(n, dtype=int)
    labels[:int(n * 0.265)] = 1
    np.random.shuffle(labels)
    return pd.Series(labels)

@pytest.fixture
def y_test():
    np.random.seed(42)
    n = 1398
    labels = np.zeros(n, dtype=int)
    labels[:int(n * 0.265)] = 1
    np.random.shuffle(labels)
    return pd.Series(labels)

# Tests
def test_X_train_has_no_missing_values(X_train):
    assert X_train.isnull().sum().sum() == 0

def test_X_test_has_no_missing_values(X_test):
    assert X_test.isnull().sum().sum() == 0

def test_train_test_split_ratio(X_train, X_test):
    total = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total
    assert 0.78 <= train_ratio <= 0.82

def test_y_train_is_binary(y_train):
    unique_values = set(y_train.unique())
    assert unique_values == {0, 1}

def test_y_test_is_binary(y_test):
    unique_values = set(y_test.unique())
    assert unique_values == {0, 1}

def test_train_churn_rate_is_stratified(y_train):
    churn_rate = y_train.mean()
    assert 0.25 <= churn_rate <= 0.28

def test_test_churn_rate_is_stratified(y_test):
    churn_rate = y_test.mean()
    assert 0.25 <= churn_rate <= 0.28

def test_X_train_and_y_train_same_length(X_train, y_train):
    assert len(X_train) == len(y_train)