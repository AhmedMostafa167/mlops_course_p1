"""Weekly retrain -> evaluate -> promotion-gated register -> notify."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.sensors.filesystem import FileSensor
from airflow.sdk import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_TYPE = "xgboost"
REGISTERED_MODEL_NAME = "taxi-duration"


@task
def extract() -> str:
    from prodml.data import download_data

    return str(download_data())


@task
def validate(data_path: str) -> str:
    from prodml.data import load_data

    df = load_data(Path(data_path))
    assert len(df) > 1000, f"only {len(df)} rows in {data_path}"
    return data_path


@task(execution_timeout=timedelta(minutes=30))
def train(data_path: str) -> str:
    os.chdir(PROJECT_ROOT)  # train.py writes metrics/new_run.json relative to cwd
    from prodml.train import main as train_main

    return train_main(["--model", MODEL_TYPE])


@task
def evaluate(run_id: str) -> float:
    import mlflow
    from mlflow import MlflowClient

    from prodml.config import get_settings

    mlflow.set_tracking_uri(get_settings().MLFLOW_TRACKING_URI)
    return MlflowClient().get_run(run_id).data.metrics["mae"]


@task.branch
def decide_promotion(mae: float) -> str:
    import mlflow
    from mlflow import MlflowClient

    from prodml.config import get_settings

    mlflow.set_tracking_uri(get_settings().MLFLOW_TRACKING_URI)
    client = MlflowClient()

    production = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    production_mae = (
        client.get_run(production[0].run_id).data.metrics["mae"] if production else float("inf")
    )
    return "register" if mae <= production_mae else "skip_registration"


@task
def register(run_id: str) -> None:
    os.chdir(PROJECT_ROOT)  # promote_if_better() writes metrics/production.json relative to cwd
    import mlflow

    from prodml.registery import promote_if_better

    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    promote_if_better(version.version, metric="mae")


@task
def skip_registration(mae: float) -> None:
    print(f"Candidate MAE {mae:.4f} did not beat production; skipping registration.")


@task
def notify() -> None:
    print("train_pipeline finished: new candidate registered/promoted.")


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 8, 3),
    catchup=False,
    max_active_runs=1,  # tasks share models/model.pkl + metrics/*.json on disk
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
)
def train_pipeline():
    wait_for_new_data = FileSensor(
        task_id="wait_for_new_data",
        filepath=str(PROJECT_ROOT / "data/landing/{{ ds_nodash }}.ready"),
        fs_conn_id="fs_default",
        poke_interval=30,
        timeout=1800,
        mode="reschedule",
    )

    data_path = extract()
    validated = validate(data_path)
    run_id = train(validated)
    mae = evaluate(run_id)
    branch = decide_promotion(mae)

    do_register = register(run_id)
    do_skip = skip_registration(mae)

    wait_for_new_data >> data_path
    branch >> [do_register, do_skip]
    do_register >> notify()


train_pipeline()