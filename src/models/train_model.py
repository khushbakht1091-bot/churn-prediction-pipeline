import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "auc_roc": roc_auc_score(y_test, probabilities)
    }
    return metrics

def train_model():
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'file:///opt/airflow/mlruns'))
    mlflow.set_experiment("churn-prediction-pipeline")

    X_train = pd.read_csv("/opt/airflow/data/processed/X_train.csv")
    X_test  = pd.read_csv("/opt/airflow/data/processed/X_test.csv")
    y_train = pd.read_csv("/opt/airflow/data/processed/y_train.csv").squeeze()
    y_test  = pd.read_csv("/opt/airflow/data/processed/y_test.csv").squeeze()

    preprocessor = joblib.load("/opt/airflow/models/preprocessor.joblib")

    with mlflow.start_run(run_name="random_forest") as rf_run:
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_metric("accuracy", rf_metrics["accuracy"])
        mlflow.log_metric("precision", rf_metrics["precision"])
        mlflow.log_metric("recall", rf_metrics["recall"])
        mlflow.log_metric("f1", rf_metrics["f1"])
        mlflow.log_metric("auc_roc", rf_metrics["auc_roc"])
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        print("Random Forest metrics:", rf_metrics)

    with mlflow.start_run(run_name="logistic_regression") as lr_run:
        lr_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        lr_model.fit(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_test, y_test)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", lr_metrics["accuracy"])
        mlflow.log_metric("precision", lr_metrics["precision"])
        mlflow.log_metric("recall", lr_metrics["recall"])
        mlflow.log_metric("f1", lr_metrics["f1"])
        mlflow.log_metric("auc_roc", lr_metrics["auc_roc"])
        mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
        print("Logistic Regression metrics:", lr_metrics)

    if lr_metrics["auc_roc"] >= rf_metrics["auc_roc"]:
        print("Winner: Logistic Regression")
        return lr_run.info.run_id, "logistic_regression_model"
    else:
        print("Winner: Random Forest")
        return rf_run.info.run_id, "random_forest_model"