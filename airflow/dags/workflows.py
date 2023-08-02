# airflow/dags/workflows.py
from pathlib import Path

from great_expectations_provider.operators.great_expectations import (
    GreatExpectationsOperator,
)

from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from config import config
from IDSAnoShift import main

GE_ROOT_DIR = Path(config.BASE_DIR, "tests", "great_expectations")

# Default DAG args
default_args = {
    "owner": "airflow",
    "catch_up": False,
}


@dag(
    dag_id="DataValidationDag",
    description="Validate Data",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["ML"],
)
def data_validation():
    extract = PythonOperator(
        task_id="extract",
        python_callable=main.extract_data,
        op_kwargs={},
    )
    validate = GreatExpectationsOperator(
        task_id="validate",
        checkpoint_name="traffic_data",
        data_context_root_dir=GE_ROOT_DIR,
        fail_task_on_validation_failure=True,
    )
    extract >> validate


# Run DAG
ml = data_validation()
