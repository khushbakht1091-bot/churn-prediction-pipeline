# Architecture Documentation

## Overview

This project solves a real business problem: predicting which telecom customers 
are likely to cancel their service before they do. Customer churn is expensive — 
acquiring a new customer costs significantly more than retaining an existing one. 
Early identification of high-risk customers enables targeted retention campaigns.

The solution is a production-grade ML pipeline built on the IBM Telco Customer 
Churn dataset (7,032 customers, 21 features). It covers the full MLOps lifecycle:
data ingestion, validation, feature engineering, model training, experiment 
tracking, model registry, REST API serving, automated retraining, CI/CD, and 
load testing.

The system is designed to run entirely in Docker, with all services orchestrated 
via Docker Compose. No manual steps are required to reproduce the pipeline.

## Components

### PostgreSQL
Serves as the raw data store. The IBM Telco CSV is ingested into a `raw_churn_data` 
table on pipeline startup. Using a database rather than reading the CSV directly 
prepares the pipeline for a real production environment where data arrives 
continuously from transactional systems, not as a static file.

### Apache Airflow
Orchestrates the pipeline as a DAG with explicit task dependencies:
`ingest_data → validate_data → feature_engineer → train_model → register_model`

Airflow was chosen over cron jobs because it provides task dependency management, 
automatic retries on failure, and a UI for monitoring pipeline runs. Each task is 
a discrete unit that can fail and retry independently without rerunning the entire 
pipeline.

### Great Expectations
Validates the raw data before it enters the feature engineering step. Enforces 16 
expectations covering null rates, value ranges, and categorical constraints. If 
validation fails, the pipeline halts before training — preventing a model from 
being trained on corrupted data.

### scikit-learn Pipeline
Encapsulates all feature engineering as a single serializable object. Categorical 
features are one-hot encoded, numeric features are scaled. Wrapping these steps in 
a Pipeline ensures the same transformations are applied identically during training 
and serving — eliminating training/serving skew.

### MLflow
Handles two responsibilities: experiment tracking and model registry. Every 
training run logs hyperparameters, AUC-ROC, and the model artifact. The registry 
maintains Production and Archived stages, enabling the champion/challenger 
retraining pattern.

### FastAPI
Serves predictions as a REST API. Accepts a JSON payload of 19 customer features, 
validates all inputs via Pydantic, transforms them using the fitted preprocessor, 
and returns a churn probability and risk label. Runs on port 8000.

### Redis
Required by Airflow's Celery executor as a message broker between the scheduler 
and worker processes. Not directly part of the ML pipeline.

### CI/CD — GitHub Actions
Every push to main triggers automated tests followed by a Docker image build and 
push to Docker Hub. This ensures the production image always reflects the latest 
passing code.


## Data Flow

The following describes how data moves through the system from raw input to 
prediction.

### Training Path

1. **Ingestion** — The raw IBM Telco CSV is read and inserted into PostgreSQL 
   (`raw_churn_data` table) by the Airflow `ingest_data` task.

2. **Validation** — Great Expectations runs 16 expectations against the raw table. 
   If any expectation fails, the DAG halts and no model is trained.

3. **Feature Engineering** — The validated data is pulled from PostgreSQL, passed 
   through the scikit-learn ColumnTransformer (one-hot encoding for categoricals, 
   standard scaling for numerics), and the fitted preprocessor is saved to 
   `models/preprocessor.joblib`.

4. **Training** — Logistic Regression is trained on the transformed features. 
   AUC-ROC is logged to MLflow along with the model artifact.

5. **Registration** — The trained model is registered in the MLflow Model Registry 
   as `churn-best-model` and promoted to Production stage.

### Serving Path

1. **Startup** — FastAPI lifespan loads the Production model from MLflow registry 
   and the preprocessor from disk into memory. This happens once at startup, not 
   on every request.

2. **Request** — A POST request arrives at `/predict` with 19 customer features 
   as JSON.

3. **Validation** — Pydantic validates all 19 fields before any ML code runs. 
   Invalid inputs are rejected with a 422 response without touching the model.

4. **Transformation** — The validated input is converted to a DataFrame and passed 
   through the fitted preprocessor.

5. **Inference** — The transformed input is scored by the model. 
   `predict_proba()[0][1]` returns the probability of churn (class 1).

6. **Response** — The probability and risk label are returned as JSON.

### Retraining Path

1. **Scheduled trigger** — The retraining DAG runs weekly via Airflow scheduler.

2. **Champion AUC retrieval** — The current Production model's AUC-ROC is fetched 
   from MLflow.

3. **Challenger training** — A new model is trained on the full current dataset.

4. **Comparison** — If challenger AUC exceeds champion AUC, the challenger is 
   promoted to Production and the previous champion is archived. Otherwise the 
   existing model is retained.


## Design Decisions

### Why Logistic Regression over Random Forest
Initial experiments compared Logistic Regression and Random Forest on AUC-ROC. 
Random Forest achieved marginally higher AUC but Logistic Regression was selected 
based on recall priority — in a churn context, missing a customer who will churn 
(false negative) is more costly than incorrectly flagging one who won't (false 
positive). Logistic Regression also offers faster inference and interpretable 
coefficients, which matters for explaining predictions to business stakeholders.

### Why the preprocessor is saved separately from the model
MLflow stores the model artifact. The preprocessor is saved separately as 
`models/preprocessor.joblib`. This is a known architectural simplification — 
ideally both would be versioned together in MLflow to prevent version mismatch. 
The current approach works for a single-model system but would not scale to 
multiple models with different preprocessing requirements.

### Why Pydantic validation covers all 19 fields
The preprocessor was trained on specific categorical values from the IBM Telco 
dataset. An unseen category — for example, a new payment method not present in 
training data — would cause the one-hot encoder to either raise an error or 
silently produce incorrect features. Validating all categorical inputs at the API 
layer prevents this failure mode before it reaches the ML pipeline.

### Why Docker Compose over individual containers
All eight services share a Docker network, allowing them to communicate by service 
name rather than IP address. This makes the stack reproducible on any machine 
with Docker installed without environment-specific configuration.

### Why champion/challenger for retraining
Automatic promotion of every retrained model introduces risk — a model trained on 
a bad data snapshot could silently replace a well-performing champion. The 
champion/challenger pattern only promotes a new model if it demonstrably 
outperforms the current Production model on AUC-ROC, making retraining safe to 
automate.


## Known Limitations

### No model monitoring
The current system has no mechanism to detect model degradation in production.
If the distribution of incoming customer features drifts from the training data,
predictions will silently become less accurate. Prometheus metrics and data drift
detection are not yet implemented.

### Preprocessor versioning is decoupled from model versioning
The preprocessor is saved as a local file rather than tracked alongside the model
in MLflow. If the model is retrained with a different preprocessing strategy, the
file must be manually updated. A mismatch between model version and preprocessor
version would produce incorrect predictions without any error being raised.

### No authentication on the API
The /predict endpoint accepts requests from any caller without authentication.
In production this would require API key validation or OAuth2 to prevent
unauthorized use and enable per-client rate limiting.

### No cloud deployment
The system is fully containerized and ready for deployment but currently runs
locally only. Docker Compose translates directly to a cloud environment with
minimal configuration changes.

### Full stack requires 8GB+ RAM
The eight-service Docker Compose stack exceeds the current development machine's
available memory. ML task execution inside Airflow is memory-constrained on this
hardware.

### Static training dataset
The pipeline ingests a fixed CSV file rather than a live data source. In a real
production system, new customer records would arrive continuously and the
retraining DAG would query a date-bounded window of recent data rather than the
full static dataset.