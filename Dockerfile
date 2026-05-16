FROM apache/airflow:2.9.1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    great-expectations==1.17.1 \
    scikit-learn \
    mlflow \
    joblib \
    pandas