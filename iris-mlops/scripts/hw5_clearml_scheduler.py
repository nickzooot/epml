from __future__ import annotations

import argparse
import os
import sys

from clearml.automation.scheduler import TaskScheduler

from clearml import Task


def _require_env() -> None:
    required = [
        "CLEARML_API_HOST",
        "CLEARML_WEB_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "ClearML client is not configured. Missing env vars: "
            f"{missing_str}. See `iris-mlops/clearml/.env.example`."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW5: schedule ClearML pipeline runs (UTC)")
    parser.add_argument(
        "--every-minutes", type=int, default=60, help="Run pipeline every N minutes"
    )
    parser.add_argument(
        "--execute-immediately",
        action="store_true",
        help="Run once immediately on start",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run once and exit (no scheduler loop)",
    )
    parser.add_argument("--project", default="iris-mlops-hw5")
    return parser.parse_args()


def _run_pipeline_once(*, project: str) -> None:
    from hw5_clearml_pipeline import main as pipeline_main

    argv_backup = list(sys.argv)
    try:
        sys.argv = [argv_backup[0], "--project", project, "--run-locally"]
        pipeline_main()
    finally:
        sys.argv = argv_backup


def main() -> None:
    _require_env()
    args = _parse_args()

    Task.init(
        project_name=args.project,
        task_name="pipeline-scheduler",
        task_type=Task.TaskTypes.service,
        tags=["hw5", "scheduler"],
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
        auto_connect_arg_parser=False,
        auto_resource_monitoring=False,
    )

    if args.run_once:
        _run_pipeline_once(project=args.project)
        return

    scheduler = TaskScheduler(sync_frequency_minutes=0.2)
    scheduler.add_task(
        schedule_function=lambda: _run_pipeline_once(project=args.project),
        name=f"hw5-pipeline-every-{args.every_minutes}m",
        minute=args.every_minutes,
        execute_immediately=args.execute_immediately,
        single_instance=True,
        recurring=True,
    )
    scheduler.start()


if __name__ == "__main__":
    main()
