from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import wandb

try:
    from iris_mlops.tracking import log_file_artifact, wandb_autolog, wandb_run
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from iris_mlops.tracking import log_file_artifact, wandb_autolog, wandb_run


@dataclass(frozen=True)
class ExperimentSpec:
    model_name: str
    params: dict[str, Any]
    needs_scaling: bool = False


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def load_iris() -> tuple[pd.DataFrame, pd.Series]:
    csv_path = _project_root() / "data" / "raw" / "iris.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["target"])
        y = df["target"]
        return X, y

    from sklearn import datasets

    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def _build_estimator(spec: ExperimentSpec, *, seed: int) -> BaseEstimator:
    estimator: BaseEstimator

    if spec.model_name == "logreg":
        estimator = LogisticRegression(max_iter=500, random_state=seed, **spec.params)
    elif spec.model_name == "svc":
        estimator = SVC(probability=True, random_state=seed, **spec.params)
    elif spec.model_name == "knn":
        estimator = KNeighborsClassifier(**spec.params)
    elif spec.model_name == "dt":
        from sklearn.tree import DecisionTreeClassifier

        estimator = DecisionTreeClassifier(random_state=seed, **spec.params)
    elif spec.model_name == "rf":
        estimator = RandomForestClassifier(random_state=seed, **spec.params)
    elif spec.model_name == "extra_trees":
        estimator = ExtraTreesClassifier(random_state=seed, **spec.params)
    elif spec.model_name == "gb":
        estimator = GradientBoostingClassifier(random_state=seed, **spec.params)
    elif spec.model_name == "ada":
        estimator = AdaBoostClassifier(random_state=seed, **spec.params)
    elif spec.model_name == "gnb":
        estimator = GaussianNB(**spec.params)
    elif spec.model_name == "lda":
        estimator = LinearDiscriminantAnalysis(**spec.params)
    elif spec.model_name == "qda":
        estimator = QuadraticDiscriminantAnalysis(**spec.params)
    else:
        raise ValueError(f"Unknown model_name: {spec.model_name}")

    if not spec.needs_scaling:
        return estimator

    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


@wandb_autolog
def train_eval(
    *,
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
) -> dict[str, float]:
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if wandb.run is not None:
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_test.to_numpy(),
                    preds=y_pred,
                    class_names=class_names,
                )
            }
        )

    return metrics


def build_experiment_plan() -> list[ExperimentSpec]:
    return [
        ExperimentSpec("logreg", {"C": 0.1}, needs_scaling=True),
        ExperimentSpec("logreg", {"C": 1.0}, needs_scaling=True),
        ExperimentSpec("logreg", {"C": 10.0}, needs_scaling=True),
        ExperimentSpec("svc", {"kernel": "rbf", "C": 1.0}, needs_scaling=True),
        ExperimentSpec("svc", {"kernel": "rbf", "C": 10.0}, needs_scaling=True),
        ExperimentSpec("svc", {"kernel": "linear", "C": 1.0}, needs_scaling=True),
        ExperimentSpec("knn", {"n_neighbors": 3}, needs_scaling=True),
        ExperimentSpec("knn", {"n_neighbors": 5}, needs_scaling=True),
        ExperimentSpec("knn", {"n_neighbors": 7}, needs_scaling=True),
        ExperimentSpec("dt", {"max_depth": None}),
        ExperimentSpec("dt", {"max_depth": 3}),
        ExperimentSpec("rf", {"n_estimators": 200, "max_depth": None}),
        ExperimentSpec("extra_trees", {"n_estimators": 200, "max_depth": None}),
        ExperimentSpec("gb", {"learning_rate": 0.1}),
        ExperimentSpec("ada", {"n_estimators": 200, "learning_rate": 0.5}),
        ExperimentSpec("gnb", {}),
        ExperimentSpec("lda", {}),
        ExperimentSpec("qda", {}),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 15+ W&B experiments on Iris dataset.")
    parser.add_argument("--project", default="iris-mlops-hw3", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity (team/user). Optional.")
    parser.add_argument("--group", default="hw3-iris", help="W&B group name for runs")
    parser.add_argument(
        "--mode",
        default=None,
        help="W&B mode: online/offline/disabled. Default: auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit number of runs for quick smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = _project_root()

    X, y = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    class_names = [str(x) for x in sorted(np.unique(y))]
    run_dir = str(base_dir)

    results: list[dict[str, Any]] = []
    plan = build_experiment_plan()
    if args.max_runs is not None:
        plan = plan[: args.max_runs]

    for idx, spec in enumerate(plan, start=1):
        run_name = f"{spec.model_name}-{idx:02d}"
        config = {
            "seed": args.seed,
            "test_size": args.test_size,
            "model_name": spec.model_name,
            "model_params": spec.params,
            "needs_scaling": spec.needs_scaling,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        tags = ["hw3", "iris", spec.model_name]

        with wandb_run(
            project=args.project,
            entity=args.entity,
            name=run_name,
            group=args.group,
            job_type="train",
            tags=tags,
            config=config,
            mode=args.mode,
            run_dir=run_dir,
        ) as run:
            estimator = _build_estimator(spec, seed=args.seed)
            metrics = train_eval(
                estimator=estimator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                class_names=class_names,
            )

            model_dir = base_dir / "models" / "wandb" / run.id
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.joblib"
            joblib.dump(estimator, model_path)
            log_file_artifact(
                name=f"model-{run.id}",
                artifact_type="model",
                path=model_path,
                aliases=["latest"],
                metadata={"model_name": spec.model_name, "model_params": spec.params},
            )

            report = classification_report(
                y_test,
                estimator.predict(X_test),
                output_dict=True,
                zero_division=0,
            )
            report_path = model_dir / "classification_report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log_file_artifact(
                name=f"report-{run.id}",
                artifact_type="report",
                path=report_path,
                aliases=["latest"],
                metadata={"model_name": spec.model_name, "model_params": spec.params},
            )

            results.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "model_name": spec.model_name,
                    "needs_scaling": spec.needs_scaling,
                    **{f"param__{k}": v for k, v in spec.params.items()},
                    **metrics,
                }
            )

            print(f"[{idx:02d}/{len(plan):02d}] {run_name}: accuracy={metrics['accuracy']:.4f}")

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    report_dir = base_dir / "reports" / "hw3"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = report_dir / "wandb_experiments_summary.csv"
    results_df.to_csv(summary_csv, index=False)

    # Optional "comparison run" with a table + summary artifact
    with wandb_run(
        project=args.project,
        entity=args.entity,
        name="summary",
        group=args.group,
        job_type="compare",
        tags=["hw3", "iris", "summary"],
        config={"n_runs": len(results_df)},
        mode=args.mode,
        run_dir=run_dir,
    ) as run:
        wandb.log({"results_table": wandb.Table(dataframe=results_df)})
        log_file_artifact(
            name=f"summary-{run.id}",
            artifact_type="report",
            path=summary_csv,
            aliases=["latest"],
            metadata={"group": args.group, "project": args.project, "n_runs": len(results_df)},
        )

    print("\nTop-5 runs by accuracy:")
    print(results_df.head(5).to_string(index=False))
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()
