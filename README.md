# Churn Prediction Pipeline

An end-to-end ML pipeline to predict customer churn.

## Architecture

```mermaid
graph TD
    A[Raw CSV Data] --> B[(PostgreSQL\nchurn_db)]
    B --> C[Airflow DAG]
    C --> D[Data Validation\nGreat Expectations]
    D --> E[Feature Engineering\nscikit-learn Pipeline]
    E --> F[Model Training\nLogistic Regression]
    F --> G[(MLflow Registry\nchurn-best-model)]
    G --> H[FastAPI\nPrediction Service]
    H --> I[POST /predict\nChurn Probability]

    style A fill:#f0f0f0
    style B fill:#336791,color:#fff
    style C fill:#017CEE,color:#fff
    style D fill:#ff6b35,color:#fff
    style E fill:#f7931e,color:#fff
    style F fill:#4caf50,color:#fff
    style G fill:#0194e2,color:#fff
    style H fill:#009688,color:#fff
    style I fill:#f0f0f0
```

## Setup 
'''bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
'''
## Pipeline

![DAG Graph](docs/dag_pipeline.png)

## REST API

### Start the API server
```bash
        MLFLOW_TRACKING_URI=file:///path/to/your/mlruns python -m uvicorn src.api.main:app --reload
```

API runs at: http://127.0.0.1:8000  
Interactive docs: http://127.0.0.1:8000/docs

### Endpoints

- `GET /health` — returns model status
- `POST /predict` — returns churn probability for a customer

### Example request

```json
{
  "tenure": 2,
  "MonthlyCharges": 85.0,
  "SeniorCitizen": 0,
  "gender": "Male",
  "Partner": "No",
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
```
### Example response

```json
{
  "churn_probability": 0.6366,
  "prediction": "High Risk"
}
```

## Running the Full Stack

All services (FastAPI, MLflow, Airflow, PostgreSQL, Redis) are orchestrated with Docker Compose.

### Prerequisites

- Docker Desktop running with WSL2 backend
- At least 4GB of memory allocated to Docker

### Start everything

```bash
docker-compose up --build
```

On first run this builds the FastAPI image and pulls all required images. Subsequent runs are faster:

```bash
docker-compose up
```

### Services

| Service | URL | Description |
|---|---|---|
| FastAPI | http://localhost:8000 | Churn prediction API |
| MLflow | http://localhost:5000 | Experiment tracking and model registry |
| Airflow | http://localhost:8080 | Pipeline orchestration (admin/admin) |

### Verify the API is running

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","model":"churn-best-model","stage":"Production"}
```

### Stop everything

```bash
docker-compose down
```

To also remove volumes (resets all data):

```bash
docker-compose down -v
```