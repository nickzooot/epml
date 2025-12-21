from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _draw_box(ax: Any, *, x: float, y: float, text: str) -> None:
    box = FancyBboxPatch(
        (x - 0.16, y - 0.05),
        0.32,
        0.1,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="#1f2937",
        facecolor="#e5e7eb",
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color="#111827")


def _arrow(ax: Any, *, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, lw=1.5)
    ax.add_patch(arrow)


def render_pipeline_diagram(out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _draw_box(ax, x=0.1, y=0.5, text="get_data")
    _draw_box(ax, x=0.3, y=0.5, text="prepare")

    _draw_box(ax, x=0.55, y=0.75, text="train_logreg")
    _draw_box(ax, x=0.55, y=0.50, text="train_rf")
    _draw_box(ax, x=0.55, y=0.25, text="train_svc")

    _draw_box(ax, x=0.8, y=0.5, text="select_best")
    _draw_box(ax, x=0.95, y=0.5, text="notify")

    _arrow(ax, x1=0.18, y1=0.5, x2=0.22, y2=0.5)
    _arrow(ax, x1=0.38, y1=0.5, x2=0.47, y2=0.75)
    _arrow(ax, x1=0.38, y1=0.5, x2=0.47, y2=0.50)
    _arrow(ax, x1=0.38, y1=0.5, x2=0.47, y2=0.25)
    _arrow(ax, x1=0.63, y1=0.75, x2=0.72, y2=0.52)
    _arrow(ax, x1=0.63, y1=0.50, x2=0.72, y2=0.50)
    _arrow(ax, x1=0.63, y1=0.25, x2=0.72, y2=0.48)
    _arrow(ax, x1=0.88, y1=0.5, x2=0.91, y2=0.5)

    ax.set_title("HW4 Airflow DAG (Iris pipeline)", fontsize=12, pad=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_best_model(out_path: pathlib.Path, summary: dict[str, Any] | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_axis_off()

    if not summary:
        ax.text(0.02, 0.7, "No summary.json found yet.", fontsize=12)
        ax.text(0.02, 0.55, "Run the DAG (or scripts) first, then re-run this script.", fontsize=11)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    best_model = summary.get("best_model")
    metric = summary.get("primary_metric")
    best_value = summary.get("best_value")
    ax.text(0.02, 0.9, "HW4: Best model", fontsize=14)
    ax.text(0.02, 0.78, f"best_model = {best_model}", fontsize=12)
    ax.text(0.02, 0.68, f"{metric} = {best_value}", fontsize=12)

    rows = summary.get("candidates") or []
    table_data = []
    for row in rows:
        table_data.append([row.get("model"), row.get("accuracy"), row.get("f1_macro")])
    table = ax.table(
        cellText=table_data,
        colLabels=["model", "accuracy", "f1_macro"],
        cellLoc="center",
        loc="lower left",
        bbox=(0.02, 0.05, 0.96, 0.5),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    project_root = _project_root()
    screenshots_dir = project_root / "reports" / "screenshots"

    render_pipeline_diagram(screenshots_dir / "hw4_airflow_dag.png")

    summary_path = project_root / "reports" / "hw4" / "summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    render_best_model(screenshots_dir / "hw4_best_model.png", summary)

    print(f"[screenshots] wrote: {screenshots_dir / 'hw4_airflow_dag.png'}")
    print(f"[screenshots] wrote: {screenshots_dir / 'hw4_best_model.png'}")


if __name__ == "__main__":
    main()
