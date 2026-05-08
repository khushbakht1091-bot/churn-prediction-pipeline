import mlflow
from mlflow.tracking import MlflowClient

# Configuration 
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "churn-logistic-regression"
RUN_ID = "d31e466dda2b46029f21f5af3b310350"

# Function for Registering model
def register_model():
    """
    Registers the Logistic Regression model from a specific MLflow run
    into the Model Registry under a permanent, versioned name.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = f"runs:/{RUN_ID}/logistic_regression_model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Model registered: {registered.name}, Version: {registered.version}")
    return registered


# Function for Promoting registered model to Staging
def promote_to_staging(version, reason):
    """
    Transitions a registered model version to Staging.
    Always requires a documented reason no silent promotions.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging"
    )

    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=f"Promoted to Staging. Reason: {reason}"
    )

    print(f"Version {version} moved to Staging. Reason: {reason}")


# Function for Promoting registered model to Production
def promote_to_production(version, reason):
    """
    Transitions a registered model version to Production.
    Always requires a documented reason no silent promotions.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production"
    )

    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=f"Promoted to Production. Reason: {reason}"
    )

    print(f"Version {version} moved to Production. Reason: {reason}")


# Function for Loading the model
def load_production_model():
    """
    Loads whichever model version is currently in Production.
    The application never needs to know the version number 
    it just asks for Production and gets the right one.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/Production"

    model = mlflow.sklearn.load_model(model_uri)

    print(f"Production model loaded: {MODEL_NAME}")
    return model


# main block
if __name__ == "__main__":

    print("--- Step 1: Registering model ---")
    registered = register_model()

    print("\n--- Step 2: Promoting to Staging ---")
    promote_to_staging(
        version=registered.version,
        reason="Logistic Regression selected over Random Forest. Higher AUC-ROC (0.839 vs 0.835) and higher recall (0.551 vs 0.495). Recall is priority metric  missing a churner is more costly than a false alarm."
    )

    print("\n--- Step 3: Promoting to Production ---")
    promote_to_production(
        version=registered.version,
        reason="Validated on held-out test set. AUC-ROC 0.839, Recall 0.551. No data leakage. Stratified split confirmed. Approved for v1 production deployment."
    )

    print("\n--- Step 4: Loading production model ---")
    model = load_production_model()
    print(f"Model type: {type(model)}")
    print("Registry pipeline complete.")