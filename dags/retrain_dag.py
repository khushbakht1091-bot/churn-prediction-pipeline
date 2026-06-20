import sys
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

MODEL_NAME = 'churn-best-model'


def on_failure_callback(context):
    task_id = context['task_instance'].task_id
    dag_id = context['task_instance'].dag_id
    execution_date = context['execution_date']
    exception = context.get('exception')
    print(f"Task FAILED: {task_id} in DAG: {dag_id}")
    print(f"Execution date: {execution_date}")
    print(f"Exception: {exception}")


def get_champion_auc(**context):
    sys.path.insert(0, '/opt/airflow')
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if not versions:
        print("No production model found. Defaulting champion AUC to 0.0")
        context['ti'].xcom_push(key='champion_auc', value=0.0)
        return

    champion_version = versions[0]
    champion_run_id = champion_version.run_id
    run = client.get_run(champion_run_id)
    champion_auc = run.data.metrics.get('auc_roc', 0.0)

    print(f"Champion model: version {champion_version.version}, AUC: {champion_auc}")
    context['ti'].xcom_push(key='champion_auc', value=champion_auc)


def retrain_model(**context):
    sys.path.insert(0, '/opt/airflow')
    from src.models.train_model import train_model

    run_id, artifact_path = train_model()
    print(f"Retraining complete. Run ID: {run_id}, Artifact: {artifact_path}")

    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(key='artifact_path', value=artifact_path)


def compare_and_promote(**context):
    sys.path.insert(0, '/opt/airflow')
    from mlflow.tracking import MlflowClient
    from src.models.model_registry import register_model

    client = MlflowClient()

    champion_auc = context['ti'].xcom_pull(task_ids='get_champion_auc', key='champion_auc')
    run_id = context['ti'].xcom_pull(task_ids='retrain_model', key='run_id')
    artifact_path = context['ti'].xcom_pull(task_ids='retrain_model', key='artifact_path')

    run = client.get_run(run_id)
    challenger_auc = run.data.metrics['auc_roc']

    print(f"Champion AUC: {champion_auc}")
    print(f"Challenger AUC: {challenger_auc}")

    if challenger_auc > champion_auc:
        print("Challenger wins. Promoting to Production.")

        current_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for version in current_versions:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version.version,
                stage="Archived"
            )
            print(f"Archived previous champion: version {version.version}")

        register_model(run_id, artifact_path)
        client.set_tag(run_id, "promotion_decision", "promoted")

    else:
        print("Champion holds. Challenger was not promoted.")
        client.set_tag(run_id, "promotion_decision", "rejected")

    client.set_tag(run_id, "champion_auc", str(round(champion_auc, 4)))
    client.set_tag(run_id, "challenger_auc", str(round(challenger_auc, 4)))

    print("Comparison complete. Tags logged to MLflow.")


with DAG(
    dag_id='churn_retrain',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@weekly',
    catchup=False,
    default_args={
        'on_failure_callback': on_failure_callback,
    },
) as dag:

    get_champion = PythonOperator(
        task_id='get_champion_auc',
        python_callable=get_champion_auc,
    )

    retrain = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
    )

    compare = PythonOperator(
        task_id='compare_and_promote',
        python_callable=compare_and_promote,
    )

    get_champion >> retrain >> compare

    