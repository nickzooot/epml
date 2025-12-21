from __future__ import annotations

import pathlib

import pendulum
from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _cmd(*parts: str) -> str:
    joined = " ".join(parts)
    return f"cd '{PROJECT_ROOT}' && {joined}"


with DAG(
    dag_id="iris_hw4_pipeline",
    description="HW4: Automated ML pipeline (Airflow + Hydra) for Iris dataset",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    max_active_tasks=4,
    params={"force": False},
    tags=["hw4", "iris", "mlops"],
) as dag:
    force = "{{ dag_run.conf.get('force', params.force) if dag_run else params.force }}"

    get_data = BashOperator(
        task_id="get_data",
        bash_command=_cmd("python", "scripts/hw4_get_data.py", f"force={force}"),
    )

    prepare = BashOperator(
        task_id="prepare",
        bash_command=_cmd("python", "scripts/hw4_prepare.py", f"force={force}"),
    )

    train_logreg = BashOperator(
        task_id="train_logreg",
        bash_command=_cmd(
            "python",
            "scripts/hw4_train.py",
            "model=logreg",
            f"force={force}",
        ),
    )
    train_rf = BashOperator(
        task_id="train_rf",
        bash_command=_cmd("python", "scripts/hw4_train.py", "model=rf", f"force={force}"),
    )
    train_svc = BashOperator(
        task_id="train_svc",
        bash_command=_cmd("python", "scripts/hw4_train.py", "model=svc", f"force={force}"),
    )

    select_best = BashOperator(
        task_id="select_best",
        bash_command=_cmd("python", "scripts/hw4_select_best.py", f"force={force}"),
    )

    notify = BashOperator(
        task_id="notify",
        bash_command=_cmd("python", "scripts/hw4_notify.py"),
    )

    get_data >> prepare >> [train_logreg, train_rf, train_svc] >> select_best >> notify
