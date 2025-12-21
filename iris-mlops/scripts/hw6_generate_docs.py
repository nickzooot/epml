from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: pathlib.Path
    reports_dir: pathlib.Path
    screenshots_dir: pathlib.Path
    docs_dir: pathlib.Path
    docs_reports_dir: pathlib.Path
    docs_assets_dir: pathlib.Path
    docs_screenshots_dir: pathlib.Path
    docs_generated_dir: pathlib.Path


def _paths() -> Paths:
    root = _project_root()
    docs_dir = root / "docs"
    return Paths(
        root=root,
        reports_dir=root / "reports",
        screenshots_dir=root / "reports" / "screenshots",
        docs_dir=docs_dir,
        docs_reports_dir=docs_dir / "reports",
        docs_assets_dir=docs_dir / "assets",
        docs_screenshots_dir=docs_dir / "assets" / "screenshots",
        docs_generated_dir=docs_dir / "assets" / "generated",
    )


def _run(cmd: list[str], *, cwd: pathlib.Path) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}")
    subprocess.run(cmd, cwd=str(cwd), check=True)  # noqa: S603


def _copy_report_md(*, src: pathlib.Path, dst: pathlib.Path) -> None:
    text = src.read_text(encoding="utf-8")
    # Rewrite screenshots relative paths: `screenshots/...` -> `../assets/screenshots/...`
    # Covers both `(screenshots/...)` and `(./screenshots/...)` patterns in Markdown links/images.
    text = re.sub(r"\(\.?(?:/)?screenshots/", "(../assets/screenshots/", text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def _sync_reports(paths: Paths) -> None:
    mapping = {
        paths.reports_dir / "hw3_wandb.md": paths.docs_reports_dir / "hw3_wandb.md",
        paths.reports_dir / "hw4_airflow.md": paths.docs_reports_dir / "hw4_airflow.md",
        paths.reports_dir / "hw5_clearml.md": paths.docs_reports_dir / "hw5_clearml.md",
        paths.reports_dir / "hw6_docs.md": paths.docs_reports_dir / "hw6_docs.md",
    }
    for src, dst in mapping.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing report file: {src}")
        _copy_report_md(src=src, dst=dst)


def _sync_screenshots(paths: Paths) -> int:
    paths.docs_screenshots_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(paths.screenshots_dir.glob("*.png")):
        dst = paths.docs_screenshots_dir / src.name
        dst.write_bytes(src.read_bytes())
        n += 1
    return n


def _ensure_hw4_outputs(paths: Paths) -> None:
    # Generates (or refreshes) local HW4 artifacts used for the generated report (fast: Iris dataset).
    hw4_reports_dir = paths.reports_dir / "hw4"
    expected = [
        hw4_reports_dir / "metrics_logreg.json",
        hw4_reports_dir / "metrics_rf.json",
        hw4_reports_dir / "metrics_svc.json",
        hw4_reports_dir / "summary.json",
    ]
    if all(p.exists() for p in expected):
        return

    _run([sys.executable, "scripts/hw4_get_data.py"], cwd=paths.root)
    _run([sys.executable, "scripts/hw4_prepare.py"], cwd=paths.root)
    _run([sys.executable, "scripts/hw4_train.py", "model=logreg"], cwd=paths.root)
    _run([sys.executable, "scripts/hw4_train.py", "model=rf"], cwd=paths.root)
    _run([sys.executable, "scripts/hw4_train.py", "model=svc"], cwd=paths.root)
    _run([sys.executable, "scripts/hw4_select_best.py"], cwd=paths.root)


def _plot_bar(
    *,
    title: str,
    labels: list[str],
    values: list[float],
    out_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.bar(labels, values, color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", labelrotation=35)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _render_hw3_section(paths: Paths) -> tuple[str, pathlib.Path]:
    csv_path = paths.reports_dir / "hw3" / "wandb_experiments_summary.csv"
    if not csv_path.exists():
        md = "\n".join(
            [
                "## HW3 (W&B): summary",
                "",
                "HW3 summary CSV is not available in the repository by default.",
                "",
                "To generate it locally (offline mode works without an account):",
                "```bash",
                "cd iris-mlops",
                "WANDB_SILENT=true python scripts/wandb_experiments.py --mode offline",
                "```",
                "",
            ]
        )
        placeholder = paths.docs_generated_dir / "hw3_wandb_top_accuracy.png"
        return md, placeholder

    df = pd.read_csv(csv_path).sort_values("accuracy", ascending=False)
    top = df.head(10).copy()
    plot_path = paths.docs_generated_dir / "hw3_wandb_top_accuracy.png"
    _plot_bar(
        title="HW3 W&B experiments (top-10 by accuracy)",
        labels=top["run_name"].astype(str).tolist(),
        values=top["accuracy"].astype(float).tolist(),
        out_path=plot_path,
    )

    table = top[
        [
            "run_name",
            "model_name",
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ]
    ].to_markdown(index=False)

    md = "\n".join(
        [
            "## HW3 (W&B): summary",
            "",
            f"- Source: `{csv_path.relative_to(paths.root)}`",
            f"- Total runs: `{len(df)}`",
            "",
            f"![HW3 W&B accuracy](../assets/generated/{plot_path.name})",
            "",
            "### Top-10 table",
            "",
            table,
            "",
        ]
    )
    return md, plot_path


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_hw4_section(*, paths: Paths, skip_hw4: bool) -> tuple[str, pathlib.Path]:
    hw4_reports_dir = paths.reports_dir / "hw4"

    metrics: list[dict[str, Any]] = []
    for model_name in ["logreg", "rf", "svc"]:
        p = hw4_reports_dir / f"metrics_{model_name}.json"
        if not p.exists():
            continue
        payload = _read_json(p)
        metrics.append({"model": model_name, **payload})

    if not metrics:
        if not skip_hw4:
            _ensure_hw4_outputs(paths)
            return _render_hw4_section(paths=paths, skip_hw4=True)

        md = "\n".join(
            [
                "## HW4 (Airflow): summary",
                "",
                "HW4 metrics are not available (missing `reports/hw4/metrics_*.json`).",
                "",
                "To generate them locally:",
                "```bash",
                "cd iris-mlops",
                "pixi install --locked",
                "pixi run hw4-run-local",
                "```",
                "",
            ]
        )
        placeholder = paths.docs_generated_dir / "hw4_airflow_accuracy.png"
        return md, placeholder

    df = pd.DataFrame(metrics).sort_values("accuracy", ascending=False)
    plot_path = paths.docs_generated_dir / "hw4_airflow_accuracy.png"
    _plot_bar(
        title="HW4 Airflow pipeline (accuracy by model)",
        labels=df["model"].astype(str).tolist(),
        values=df["accuracy"].astype(float).tolist(),
        out_path=plot_path,
    )

    summary_path = hw4_reports_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    best_model = summary.get("best_model")
    best_value = summary.get("best_value")

    table = df[["model", "accuracy", "f1_macro", "precision_macro", "recall_macro"]].to_markdown(index=False)

    md = "\n".join(
        [
            "## HW4 (Airflow): summary",
            "",
            f"- Generated from local artifacts in `{hw4_reports_dir.relative_to(paths.root)}`",
            f"- Best model: `{best_model}` (accuracy={best_value})",
            "",
            f"![HW4 accuracy](../assets/generated/{plot_path.name})",
            "",
            "### Table",
            "",
            table,
            "",
        ]
    )
    return md, plot_path


def _render_hw5_section_optional(paths: Paths) -> tuple[str, pathlib.Path | None]:
    csv_path = paths.reports_dir / "hw5" / "clearml_experiments_summary.csv"
    if not csv_path.exists():
        md = "\n".join(
            [
                "## HW5 (ClearML): summary",
                "",
                "ClearML summary CSV is not available in the repository by default (it is environment‑dependent).",
                "",
                "To generate it locally:",
                "```bash",
                "cd iris-mlops/clearml",
                "docker compose --env-file .env.example -f docker-compose.yml up -d",
                "cd ..",
                "export CLEARML_API_HOST=http://localhost:8008",
                "export CLEARML_WEB_HOST=http://localhost:8080",
                "export CLEARML_FILES_HOST=http://localhost:8081",
                "export CLEARML_API_ACCESS_KEY=hw5_local_access",
                "export CLEARML_API_SECRET_KEY=hw5_local_secret",
                "pixi install --locked",
                "pixi run hw5-experiments",
                "pixi run hw5-compare",
                "```",
                "",
            ]
        )
        return md, None

    df = pd.read_csv(csv_path).sort_values("accuracy", ascending=False)
    top = df.head(10).copy()
    plot_path = paths.docs_generated_dir / "hw5_clearml_top_accuracy.png"
    _plot_bar(
        title="HW5 ClearML experiments (top-10 by accuracy)",
        labels=top["name"].astype(str).tolist(),
        values=top["accuracy"].astype(float).tolist(),
        out_path=plot_path,
    )

    table = top[["name", "status", "accuracy", "web_url"]].to_markdown(index=False)
    md = "\n".join(
        [
            "## HW5 (ClearML): summary",
            "",
            f"- Source: `{csv_path.relative_to(paths.root)}`",
            f"- Total tasks: `{len(df)}`",
            "",
            f"![HW5 ClearML accuracy](../assets/generated/{plot_path.name})",
            "",
            "### Top-10 table",
            "",
            table,
            "",
        ]
    )
    return md, plot_path


def _write_generated_report(*, paths: Paths, skip_hw4: bool) -> pathlib.Path:
    parts: list[str] = ["# Generated experiment summary", ""]

    hw3_md, _ = _render_hw3_section(paths)
    parts.append(hw3_md)

    hw4_md, _ = _render_hw4_section(paths=paths, skip_hw4=skip_hw4)
    parts.append(hw4_md)

    hw5_md, _ = _render_hw5_section_optional(paths)
    parts.append(hw5_md)

    out_path = paths.docs_reports_dir / "experiments_generated.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW6: generate MkDocs inputs and reports")
    parser.add_argument(
        "--skip-hw4",
        action="store_true",
        help="Do not run HW4 scripts for generating summary artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = _paths()

    paths.docs_generated_dir.mkdir(parents=True, exist_ok=True)
    paths.docs_reports_dir.mkdir(parents=True, exist_ok=True)

    _sync_reports(paths)
    copied = _sync_screenshots(paths)
    print(f"[hw6] synced screenshots: {copied}")

    if args.skip_hw4:
        print("[hw6] skip HW4 generation")
    else:
        _ensure_hw4_outputs(paths)

    out_path = _write_generated_report(paths=paths, skip_hw4=bool(args.skip_hw4))
    print("[hw6] generated:", out_path)


if __name__ == "__main__":
    main()
