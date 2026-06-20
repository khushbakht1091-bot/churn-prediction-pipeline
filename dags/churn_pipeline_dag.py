import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable


def on_failure_callback(context):
    task_id = context['task_instance'].task_id
    dag_id = context['task_instance'].dag_id
    execution_date = context['execution_date']
    exception = context.get('exception')
    print(f"Task FAILED: {task_id} in DAG: {dag_id}")
    print(f"Execution date: {execution_date}")
    print(f"Exception: {exception}")


def ingest_data_task(**context):
    sys.path.insert(0, '/opt/airflow')
    csv_path = Variable.get("churn_csv_path")
    from src.data.ingest import ingest_data
    ingest_data(csv_path)


def validate_data_task(**context):
    sys.path.insert(0, '/opt/airflow')
    csv_path = Variable.get("churn_csv_path")
    import pandas as pd
    from src.data.validate_data import validate_data
    df = pd.read_csv(csv_path)
    validate_data(df)


def feature_engineer_task(**context):
    sys.path.insert(0, '/opt/airflow')
    csv_path = Variable.get("churn_csv_path")
    import pandas as pd
    from src.features.build_features import build_features
    df = pd.read_csv(csv_path)
    build_features(df)


def train_model_task(**context):
    sys.path.insert(0, '/opt/airflow')
    from src.models.train_model import train_model
    run_id, artifact_path = train_model()
    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(key='artifact_path', value=artifact_path)


def register_model_task(**context):
    sys.path.insert(0, '/opt/airflow')
    run_id = context['ti'].xcom_pull(task_ids='train_model', key='run_id')
    artifact_path = context['ti'].xcom_pull(task_ids='train_model', key='artifact_path')
    from src.models.model_registry import register_model
    register_model(run_id, artifact_path)


with DAG(
    dag_id='churn_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@weekly',
    catchup=False,
    default_args={
        'on_failure_callback': on_failure_callback,
    },
) as dag:

    ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data_task,
    )

    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_task,
    )

    engineer = PythonOperator(
        task_id='feature_engineer',
        python_callable=feature_engineer_task,
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )

    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model_task,
    )

    ingest >> validate >> engineer >> train >> register