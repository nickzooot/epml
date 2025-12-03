import pathlib
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def main() -> None:
    tracking_uri = "sqlite:///mlruns.db"
    artifact_location = "mlruns"
    experiment_name = "iris-baseline"
    model_name = "iris-classifier"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Simple baseline model
    clf = LogisticRegression(max_iter=200)

    with mlflow.start_run(run_name="iris-logreg") as run:
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_params({"model": "logreg", "max_iter": 200})
        mlflow.log_metrics({"accuracy": acc})

        # Save and register model
        model_path = pathlib.Path("models") / "logreg"
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X.iloc[:5],
        )

        # Add run metadata for quick diffing
        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag("data_version", "dvc:get_data")

        print(
            f"Run {run.info.run_id} finished. Accuracy={acc:.4f}. "
            f"Registered as {model_name}."
        )


if __name__ == "__main__":
    main()
