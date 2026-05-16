from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
import sys
import os

default_args = {
    'owner': 'khushbakht',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


project_dir = '/opt/airflow'

def run_validate_data():
    result = subprocess.run(
        [sys.executable, os.path.join(project_dir, 'src', 'data', 'validate_data.py')],
        capture_output=True,
        text=True,
        cwd=project_dir
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def run_feature_engineer():
    result = subprocess.run(
        [sys.executable, os.path.join(project_dir, 'src', 'features', 'build_features.py')],
        capture_output=True,
        text=True,
        cwd=project_dir
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def run_train_model():
    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = 'file:///opt/airflow/mlruns'
    result = subprocess.run(
        [sys.executable, os.path.join(project_dir, 'src', 'models', 'train_model.py')],
        capture_output=True,
        text=True,
        cwd=project_dir,
        env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)
    
with DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='Automated churn prediction pipeline',
    schedule_interval=None,
    catchup=False,
) as dag:

    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=run_validate_data,
    )

    feature_engineer = PythonOperator(
        task_id='feature_engineer',
        python_callable=run_feature_engineer,
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_train_model,
    )

    validate_data >> feature_engineer >> train_model


