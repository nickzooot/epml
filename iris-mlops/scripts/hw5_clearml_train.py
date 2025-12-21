from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

import hydra
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from clearml import OutputModel, Task


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


def _validate(cfg: DictConfig) -> None:
    test_size = float(cfg.data.test_size)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"data.test_size must be in (0, 1), got: {test_size}")

    model_name = str(cfg.model.name)
    if model_name not in {"logreg", "rf", "svc"}:
        raise ValueError(f"Unknown model.name: {model_name}")


def _load_iris(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    root = _project_root()
    csv_path = root / pathlib.Path(str(cfg.paths.raw)) / str(cfg.data.filename)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        target = str(cfg.data.target_col)
        X = df.drop(columns=[target])
        y = df[target]
        return X, y

    from sklearn import datasets

    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def _build_estimator(cfg: DictConfig) -> BaseEstimator:
    model_name = str(cfg.model.name)
    needs_scaling = bool(cfg.model.needs_scaling)

    params_raw = OmegaConf.to_container(cfg.model.params, resolve=True) or {}
    if not isinstance(params_raw, dict):
        raise TypeError("model.params must be a mapping")
    params: dict[str, Any] = {str(k): v for k, v in params_raw.items()}

    if model_name == "logreg":
        estimator: BaseEstimator = LogisticRegression(random_state=int(cfg.seed), **params)
    elif model_name == "rf":
        estimator = RandomForestClassifier(random_state=int(cfg.seed), **params)
    elif model_name == "svc":
        estimator = SVC(probability=True, random_state=int(cfg.seed), **params)
    else:
        raise ValueError(f"Unknown model.name: {model_name}")

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


@hydra.main(version_base=None, config_path="../configs/hw5", config_name="config")
def main(cfg: DictConfig) -> None:
    _require_env()
    _validate(cfg)

    model_name = str(cfg.model.name)
    group = str(cfg.clearml.group)
    tags = [str(x) for x in list(cfg.clearml.tags)] + [model_name, f"group:{group}"]

    task_name = f"{model_name}-{int(time.time())}"
    task = Task.init(
        project_name=str(cfg.clearml.project),
        task_name=task_name,
        task_type=Task.TaskTypes.training,
        tags=tags,
        reuse_last_task_id=False,
        auto_connect_arg_parser=False,
        auto_connect_frameworks=True,
    )

    task.connect(OmegaConf.to_container(cfg, resolve=True), name="cfg")
    logger = task.get_logger()

    X, y = _load_iris(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.data.test_size),
        random_state=int(cfg.seed),
        stratify=y,
    )

    estimator = _build_estimator(cfg)
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

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    task.upload_artifact("metrics", metrics)
    task.upload_artifact("classification_report", report)

    root = _project_root()
    model_dir = root / pathlib.Path(str(cfg.paths.models)) / model_name / task.id
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(estimator, model_path)

    output_model = OutputModel(
        task=task,
        name=str(cfg.clearml.output_model_name),
        tags=tags,
        framework="scikit-learn",
        comment="HW5: scikit-learn classifier on Iris dataset",
    )
    output_model.set_metadata("model_name", model_name)
    output_model.set_metadata("seed", str(int(cfg.seed)))
    output_model.set_metadata("test_size", str(float(cfg.data.test_size)))
    output_model.set_metadata("accuracy", f"{metrics['accuracy']:.6f}")
    output_model.set_metadata("metrics_json", json.dumps(metrics, ensure_ascii=False))
    output_model.update_weights(
        weights_filename=str(model_path),
        auto_delete_file=False,
        iteration=int(time.time()),
    )

    reports_dir = root / pathlib.Path(str(cfg.paths.reports))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"metrics_{task.id}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[hw5] task_id={task.id} model={model_name} accuracy={metrics['accuracy']:.4f}")
    task.close()


if __name__ == "__main__":
    main()
