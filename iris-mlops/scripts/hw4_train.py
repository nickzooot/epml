from __future__ import annotations

import json
import pathlib
from typing import Any

import hydra
import joblib
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _validate(cfg: DictConfig) -> None:
    test_size = float(cfg.data.test_size)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"data.test_size must be in (0, 1), got: {test_size}")

    model_name = str(cfg.model.name)
    if model_name not in {"logreg", "rf", "svc"}:
        raise ValueError(f"Unknown model.name: {model_name}")


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


@hydra.main(version_base=None, config_path="../configs/hw4", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate(cfg)
    project_root = _project_root()

    processed_dir = project_root / pathlib.Path(str(cfg.paths.processed))
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Prepared dataset is missing: {processed_dir} (run prepare first)")

    model_name = str(cfg.model.name)
    model_dir = project_root / pathlib.Path(str(cfg.paths.models)) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"

    reports_dir = project_root / pathlib.Path(str(cfg.paths.reports))
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / f"metrics_{model_name}.json"
    report_path = reports_dir / f"classification_report_{model_name}.json"

    if (
        model_path.exists()
        and metrics_path.exists()
        and report_path.exists()
        and not bool(cfg.force)
    ):
        print(f"[train:{model_name}] cache hit -> skip: {model_path}")
        return

    target_col = str(cfg.data.target_col)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    estimator = _build_estimator(cfg)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    joblib.dump(estimator, model_path)

    print(f"[train:{model_name}] saved: {model_path}")
    print(f"[train:{model_name}] metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
