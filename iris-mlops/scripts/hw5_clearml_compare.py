from __future__ import annotations

import argparse
import pathlib
from typing import Any

import pandas as pd

from clearml import Task


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ClearML HW5 experiments")
    parser.add_argument("--project", default="iris-mlops-hw5")
    parser.add_argument("--group", default="hw5-exp", help="Filter by tag: group:<value>")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument(
        "--include-non-completed",
        action="store_true",
        help="Include failed/aborted/running tasks in the table",
    )
    return parser.parse_args()


def _get_metric(task: Task, *, title: str, series: str) -> float | None:
    try:
        last = task.get_last_scalar_metrics()
    except Exception:
        return None
    try:
        return float(last[title][series]["last"])
    except Exception:
        return None


def main() -> None:
    args = _parse_args()
    tags = ["hw5", "iris", f"group:{args.group}"]
    tasks = Task.get_tasks(project_name=args.project, tags=tags)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        status = str(task.status).lower()
        if not args.include_non_completed and status != "completed":
            continue

        acc = _get_metric(task, title="metrics", series="accuracy")
        if acc is None:
            continue
        if args.min_accuracy is not None and acc < args.min_accuracy:
            continue

        rows.append(
            {
                "task_id": task.id,
                "name": task.name,
                "status": status,
                "accuracy": acc,
                "web_url": task.get_output_log_web_page(),
            }
        )

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    if df.empty:
        print("[hw5] No tasks found. Did you run `hw5_clearml_experiments.py`?")
        return

    print(df.head(args.top_k).to_string(index=False))

    out_dir = _project_root() / "reports" / "hw5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clearml_experiments_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[hw5] saved: {out_path}")


if __name__ == "__main__":
    main()
