import argparse
import json
import os
import pathlib
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml.automation.controller import PipelineController
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from clearml import Model, OutputModel, Task


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


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


def _load_iris_from_repo(
    *, project_root: pathlib.Path, raw_dir: str, filename: str, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    csv_path = project_root / pathlib.Path(raw_dir) / filename
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y

    from sklearn import datasets

    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def _build_estimator(
    *,
    model_name: str,
    params: dict[str, Any],
    needs_scaling: bool,
    seed: int,
) -> BaseEstimator:
    if model_name == "logreg":
        estimator: BaseEstimator = LogisticRegression(random_state=seed, **params)
    elif model_name == "rf":
        estimator = RandomForestClassifier(random_state=seed, **params)
    elif model_name == "svc":
        estimator = SVC(probability=True, random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if not needs_scaling:
        return estimator

    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _plot_confusion(cm: np.ndarray, class_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def train_step(
    *,
    model_name: str,
    params: dict[str, Any],
    needs_scaling: bool,
    seed: int,
    test_size: float,
    raw_dir: str,
    data_filename: str,
    target_col: str,
    project_root: str | None,
    output_model_name: str,
    group: str,
    models_dir: str,
) -> dict[str, Any]:
    task = Task.current_task()
    tags = ["hw5", "iris", model_name, f"group:{group}"]

    if task is None:
        task = Task.init(
            project_name="iris-mlops-hw5",
            task_name=f"train-{model_name}",
            task_type=Task.TaskTypes.training,
            tags=tags,
            reuse_last_task_id=False,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=True,
        )
    else:
        task.add_tags(tags)

    task.connect(
        {
            "model_name": model_name,
            "params": params,
            "needs_scaling": needs_scaling,
            "seed": seed,
            "test_size": test_size,
        },
        name="cfg",
    )

    logger = task.get_logger()

    root = pathlib.Path(project_root) if project_root else pathlib.Path.cwd()
    X, y = _load_iris_from_repo(
        project_root=root, raw_dir=raw_dir, filename=data_filename, target_col=target_col
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    estimator = _build_estimator(
        model_name=model_name, params=params, needs_scaling=needs_scaling, seed=seed
    )
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    for k, v in metrics.items():
        logger.report_scalar(title="metrics", series=k, value=float(v), iteration=0)

    cm = confusion_matrix(y_test, y_pred)
    fig = _plot_confusion(cm, class_names=[str(x) for x in sorted(np.unique(y))])
    logger.report_matplotlib_figure(
        title="confusion_matrix",
        series=model_name,
        figure=fig,
        iteration=0,
        report_image=True,
    )
    plt.close(fig)

    task.upload_artifact("metrics", metrics)

    model_dir = root / pathlib.Path(models_dir) / "pipeline" / group / model_name / task.id
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(estimator, model_path)

    output_model = OutputModel(
        task=task,
        name=output_model_name,
        tags=tags,
        framework="scikit-learn",
        comment="HW5 (pipeline): scikit-learn classifier on Iris dataset",
    )
    output_model.set_metadata("model_name", model_name)
    output_model.set_metadata("seed", str(seed))
    output_model.set_metadata("test_size", str(test_size))
    output_model.set_metadata("accuracy", f"{metrics['accuracy']:.6f}")
    output_model.set_metadata("metrics_json", json.dumps(metrics, ensure_ascii=False))
    output_model.update_weights(
        weights_filename=str(model_path),
        auto_delete_file=False,
        iteration=int(time.time()),
    )

    task.flush(wait_for_uploads=True)
    return {"task_id": task.id, "model_id": output_model.id, **metrics}


def _safe_task_accuracy(task: Task) -> float | None:
    try:
        last = task.get_last_scalar_metrics()
        return float(last["metrics"]["accuracy"]["last"])
    except (KeyError, TypeError, ValueError):
        return None


def _safe_model_accuracy(model: Model) -> tuple[float | None, dict[str, Any] | None]:
    try:
        meta = model.get_all_metadata_casted()
    except Exception:
        meta = {}

    raw = meta.get("metrics_json")
    if not raw:
        return None, None

    try:
        metrics = json.loads(str(raw))
        acc = float(metrics.get("accuracy", float("-inf")))
    except (TypeError, ValueError):
        return None, None

    return acc, metrics


def select_best_step(
    *,
    project: str,
    group: str,
    output_model_name: str,
    reports_dir: str,
    project_root: str | None,
) -> dict[str, Any]:
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name=project,
            task_name="select-best",
            task_type=Task.TaskTypes.qc,
            tags=["hw5", "iris", f"group:{group}"],
            reuse_last_task_id=False,
        )

    tags = [f"group:{group}"]
    tasks = Task.get_tasks(project_name=project, tags=tags)
    rows: list[dict[str, Any]] = []
    for t in tasks:
        acc = _safe_task_accuracy(t)
        if acc is None:
            continue
        rows.append({"task_id": t.id, "task_name": t.name, "accuracy": acc, "status": t.status})

    if not rows:
        raise RuntimeError("No training tasks found for the pipeline run")

    best_task = max(rows, key=lambda r: float(r["accuracy"]))

    models = Model.query_models(
        project_name=project,
        model_name=output_model_name,
        tags=tags,
        include_archived=False,
        max_results=200,
    )
    best_model_id: str | None = None
    best_model_metrics: dict[str, Any] | None = None
    best_model_acc = float("-inf")
    for m in models:
        acc, metrics = _safe_model_accuracy(m)
        if acc is None or metrics is None:
            continue
        if acc <= best_model_acc:
            continue
        best_model_acc = acc
        best_model_id = m.id
        best_model_metrics = metrics

    summary = {
        "group": group,
        "best_task": best_task,
        "best_model_id": best_model_id,
        "best_model_metrics": best_model_metrics,
        "candidates": sorted(rows, key=lambda r: float(r["accuracy"]), reverse=True),
        "ts_unix": int(time.time()),
    }

    root = pathlib.Path(project_root) if project_root else pathlib.Path.cwd()
    out_dir = root / pathlib.Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"pipeline_summary_{group}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    task.upload_artifact("pipeline_summary", summary)
    task.get_logger().report_scalar(
        title="best", series="accuracy", value=float(best_task["accuracy"]), iteration=0
    )
    task.flush(wait_for_uploads=True)
    print(f"[hw5 pipeline] best_task={best_task['task_id']} accuracy={best_task['accuracy']:.4f}")
    return {"summary_path": str(summary_path)}


def notify_step(*, group: str, reports_dir: str, project_root: str | None) -> None:
    root = pathlib.Path(project_root) if project_root else pathlib.Path.cwd()
    out_dir = root / pathlib.Path(reports_dir)
    summary_path = out_dir / f"pipeline_summary_{group}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_task = summary.get("best_task") or {}
    message = (
        f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
        f"HW5 ClearML pipeline finished: group={group} "
        f"best_task={best_task.get('task_id')} accuracy={best_task.get('accuracy')}"
    )
    print(message)

    log_path = out_dir / "notifications.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")

    t = Task.current_task()
    if t is not None:
        t.upload_artifact("notification", {"message": message})
        t.flush(wait_for_uploads=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW5 ClearML pipeline (local execution)")
    parser.add_argument("--project", default="iris-mlops-hw5")
    parser.add_argument(
        "--run-locally",
        action="store_true",
        help="Run steps locally (no ClearML Agent needed)",
    )
    parser.add_argument(
        "--queue",
        default="services",
        help="Execution queue (if not running locally)",
    )
    return parser.parse_args()


def main() -> None:
    _require_env()
    args = _parse_args()

    project_root = str(_project_root())
    group = f"hw5-pipeline-{int(time.time())}"
    pipeline = PipelineController(
        name="iris-hw5-pipeline",
        project=args.project,
        version=None,
        add_pipeline_tags=True,
        abort_on_failure=True,
    )

    pipeline.add_function_step(
        name="train_logreg",
        function=train_step,
        function_kwargs={
            "model_name": "logreg",
            "params": {"C": 1.0, "max_iter": 500},
            "needs_scaling": True,
            "seed": 42,
            "test_size": 0.2,
            "raw_dir": "data/raw",
            "data_filename": "iris.csv",
            "target_col": "target",
            "project_root": project_root,
            "output_model_name": "iris-hw5-classifier",
            "group": group,
            "models_dir": "models/hw5",
        },
        function_return=[
            "task_id",
            "model_id",
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ],
        project_name=args.project,
        task_name=f"pipeline-train-logreg ({group})",
        task_type="training",
        tags=["hw5", "iris", "logreg", f"group:{group}"],
        helper_functions=[_project_root, _load_iris_from_repo, _build_estimator, _plot_confusion],
        cache_executed_step=True,
    )
    pipeline.add_function_step(
        name="train_rf",
        function=train_step,
        function_kwargs={
            "model_name": "rf",
            "params": {"n_estimators": 300, "max_depth": None},
            "needs_scaling": False,
            "seed": 42,
            "test_size": 0.2,
            "raw_dir": "data/raw",
            "data_filename": "iris.csv",
            "target_col": "target",
            "project_root": project_root,
            "output_model_name": "iris-hw5-classifier",
            "group": group,
            "models_dir": "models/hw5",
        },
        function_return=[
            "task_id",
            "model_id",
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ],
        project_name=args.project,
        task_name=f"pipeline-train-rf ({group})",
        task_type="training",
        tags=["hw5", "iris", "rf", f"group:{group}"],
        helper_functions=[_project_root, _load_iris_from_repo, _build_estimator, _plot_confusion],
        cache_executed_step=True,
    )
    pipeline.add_function_step(
        name="train_svc",
        function=train_step,
        function_kwargs={
            "model_name": "svc",
            "params": {"kernel": "rbf", "C": 3.0},
            "needs_scaling": True,
            "seed": 42,
            "test_size": 0.2,
            "raw_dir": "data/raw",
            "data_filename": "iris.csv",
            "target_col": "target",
            "project_root": project_root,
            "output_model_name": "iris-hw5-classifier",
            "group": group,
            "models_dir": "models/hw5",
        },
        function_return=[
            "task_id",
            "model_id",
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ],
        project_name=args.project,
        task_name=f"pipeline-train-svc ({group})",
        task_type="training",
        tags=["hw5", "iris", "svc", f"group:{group}"],
        helper_functions=[_project_root, _load_iris_from_repo, _build_estimator, _plot_confusion],
        cache_executed_step=True,
    )

    pipeline.add_function_step(
        name="select_best",
        function=select_best_step,
        function_kwargs={
            "project": args.project,
            "group": group,
            "output_model_name": "iris-hw5-classifier",
            "reports_dir": "reports/hw5",
            "project_root": project_root,
        },
        function_return=["summary_path"],
        project_name=args.project,
        task_name=f"pipeline-select-best ({group})",
        task_type="qc",
        tags=["hw5", "iris", "select_best", f"group:{group}"],
        helper_functions=[_project_root, _safe_task_accuracy, _safe_model_accuracy],
        parents=["train_logreg", "train_rf", "train_svc"],
    )

    pipeline.add_function_step(
        name="notify",
        function=notify_step,
        function_kwargs={
            "group": group,
            "reports_dir": "reports/hw5",
            "project_root": project_root,
        },
        project_name=args.project,
        task_name=f"pipeline-notify ({group})",
        task_type="service",
        tags=["hw5", "iris", "notify", f"group:{group}"],
        helper_functions=[_project_root],
        parents=["select_best"],
    )

    if args.run_locally:
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        pipeline.start(queue=args.queue, wait=True)

    print(f"[hw5 pipeline] finished group={group}")


if __name__ == "__main__":
    main()
