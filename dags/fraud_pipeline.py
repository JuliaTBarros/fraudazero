"""DAG do FraudaZero: ingestão, preparação, treino do baseline e recarga do modelo na API."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# O treino roda no venv /opt/train-venv (dependências separadas das do Airflow).
TRAIN = (
    "cd /opt/project && PYTHONPATH=/opt/project /opt/train-venv/bin/python -m src.train"
)

# @daily: roda sozinha na 1ª subida (catchup=False); retreinar de hora em hora só gasta RAM.
with DAG(
    dag_id="fraudazero_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["fraudazero"],
) as dag:
    ingest = BashOperator(task_id="ingest", bash_command=f"{TRAIN} ingest")
    prepare = BashOperator(task_id="prepare", bash_command=f"{TRAIN} prepare")
    baseline = BashOperator(task_id="baseline", bash_command=f"{TRAIN} baseline")
    reload_api = BashOperator(
        task_id="reload_api",
        bash_command="curl -sf -X POST http://api:8000/reload || echo 'API ainda indisponível'",
    )

    ingest >> prepare >> baseline >> reload_api
