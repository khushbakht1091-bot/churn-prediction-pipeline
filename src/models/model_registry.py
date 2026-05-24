import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "churn-best-model"

def register_model(run_id, artifact_path, model_name=MODEL_NAME):
    """
    Registers the winning model from a specific MLflow run
    into the Model Registry, then promotes it to Production.
    """
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'file:///opt/airflow/mlruns'))
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/{artifact_path}"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"Model registered: {registered.name}, Version: {registered.version}")

    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Staging"
    )
    print(f"Version {registered.version} moved to Staging")

    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Production"
    )
    print(f"Version {registered.version} moved to Production")

    return registered


def load_production_model(model_name=MODEL_NAME):
    """
    Loads whichever model version is currently in Production.
    """
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'file:///opt/airflow/mlruns'))
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Production model loaded: {model_name}")
    return model