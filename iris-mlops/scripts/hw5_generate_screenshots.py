from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

try:
    from clearml import Model
except ModuleNotFoundError:  # pragma: no cover
    Model = None  # type: ignore[assignment]


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _env_ready() -> bool:
    return all(
        os.getenv(k)
        for k in [
            "CLEARML_API_HOST",
            "CLEARML_WEB_HOST",
            "CLEARML_FILES_HOST",
            "CLEARML_API_ACCESS_KEY",
            "CLEARML_API_SECRET_KEY",
        ]
    )


def _save_placeholder(path: pathlib.Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_axis_off()
    ax.text(0.02, 0.85, title, fontsize=14, fontweight="bold")
    ax.text(0.02, 0.65, message, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _render_experiments_chart(out_path: pathlib.Path) -> None:
    root = _project_root()
    csv_path = root / "reports" / "hw5" / "clearml_experiments_summary.csv"
    if not csv_path.exists():
        _save_placeholder(
            out_path,
            "HW5 ClearML: experiments",
            f"Missing {csv_path}. Run `python scripts/hw5_clearml_experiments.py` and "
            "`python scripts/hw5_clearml_compare.py` first.",
        )
        return

    df = pd.read_csv(csv_path).sort_values("accuracy", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.bar(df["name"], df["accuracy"], color="#2563eb")
    ax.set_title("HW5 ClearML experiments (top-12 by accuracy)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", labelrotation=35)
    for i, v in enumerate(df["accuracy"].tolist()):
        ax.text(i, float(v) + 0.01, f"{float(v):.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _parse_model_accuracy(meta: dict[str, Any]) -> float | None:
    raw = meta.get("accuracy")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    raw_json = meta.get("metrics_json")
    if not raw_json:
        return None
    try:
        payload = json.loads(str(raw_json))
        return float(payload.get("accuracy"))
    except (TypeError, ValueError, KeyError):
        return None


def _render_models_table(out_path: pathlib.Path) -> None:
    if Model is None or not _env_ready():
        _save_placeholder(
            out_path,
            "HW5 ClearML: model registry",
            "ClearML client env is not configured. "
            "Export vars from `iris-mlops/clearml/.env.example`.",
        )
        return

    models = Model.query_models(
        project_name="iris-mlops-hw5",
        model_name="iris-hw5-classifier",
        tags=["hw5", "iris"],
        include_archived=False,
        max_results=20,
    )
    if not models:
        _save_placeholder(
            out_path,
            "HW5 ClearML: model registry",
            "No models found. Run `python scripts/hw5_clearml_train.py` first.",
        )
        return

    rows: list[dict[str, Any]] = []
    for m in models:
        try:
            meta = m.get_all_metadata_casted()
        except Exception:
            meta = {}
        rows.append(
            {
                "id": m.id,
                "tags": ",".join(m.tags or []),
                "accuracy": _parse_model_accuracy(meta),
            }
        )

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_axis_off()
    ax.set_title("HW5 ClearML model registry (top-10 by accuracy)", fontsize=13, pad=12)
    table = ax.table(
        cellText=df.fillna("").values,
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _render_pipeline_summary(out_path: pathlib.Path) -> None:
    root = _project_root()
    hw5_dir = root / "reports" / "hw5"
    candidates = sorted(hw5_dir.glob("pipeline_summary_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        _save_placeholder(
            out_path,
            "HW5 ClearML: pipeline",
            "No pipeline summary files found. "
            "Run `python scripts/hw5_clearml_pipeline.py --run-locally` first.",
        )
        return

    summary_path = candidates[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_task = summary.get("best_task") or {}

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_axis_off()
    ax.text(0.02, 0.9, "HW5 ClearML pipeline summary", fontsize=14, fontweight="bold")
    ax.text(0.02, 0.75, f"file: {summary_path.name}", fontsize=10)
    ax.text(0.02, 0.62, f"group: {summary.get('group')}", fontsize=11)
    ax.text(0.02, 0.50, f"best_task: {best_task.get('task_id')}", fontsize=11)
    ax.text(0.02, 0.40, f"best_accuracy: {best_task.get('accuracy')}", fontsize=11)

    candidates_rows = summary.get("candidates") or []
    df = pd.DataFrame(candidates_rows)[["task_id", "accuracy", "status"]].head(8)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns.tolist(),
        loc="lower left",
        cellLoc="left",
        bbox=(0.02, 0.05, 0.96, 0.28),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    root = _project_root()
    screenshots_dir = root / "reports" / "screenshots"

    _render_experiments_chart(screenshots_dir / "hw5_clearml_experiments.png")
    _render_models_table(screenshots_dir / "hw5_clearml_models.png")
    _render_pipeline_summary(screenshots_dir / "hw5_clearml_pipeline.png")

    print("[hw5] screenshots generated in:", screenshots_dir)


if __name__ == "__main__":
    main()
