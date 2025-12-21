from __future__ import annotations

import json
import pathlib
import shutil
import time
from typing import Any

import hydra
from omegaconf import DictConfig


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../configs/hw4", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = _project_root()
    reports_dir = project_root / pathlib.Path(str(cfg.paths.reports))
    models_dir = project_root / pathlib.Path(str(cfg.paths.models))

    primary_metric = str(cfg.metrics.primary)
    model_names = [str(x) for x in list(cfg.selection.models)]
    if not model_names:
        raise ValueError("selection.models must be a non-empty list")

    results: list[dict[str, Any]] = []
    best_model: str | None = None
    best_value = float("-inf")

    for model_name in model_names:
        metrics_path = reports_dir / f"metrics_{model_name}.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for model '{model_name}': {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if primary_metric not in metrics:
            raise KeyError(f"Metric '{primary_metric}' not found in {metrics_path}")

        value = float(metrics[primary_metric])
        results.append({"model": model_name, **metrics})
        if value > best_value:
            best_model = model_name
            best_value = value

    if best_model is None:
        raise RuntimeError("Failed to select best model")

    best_model_path = models_dir / best_model / "model.joblib"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Missing model artifact for '{best_model}': {best_model_path}")

    best_out_dir = models_dir / "best"
    best_out_dir.mkdir(parents=True, exist_ok=True)
    best_out_path = best_out_dir / "model.joblib"
    shutil.copy2(best_model_path, best_out_path)

    summary = {
        "best_model": best_model,
        "primary_metric": primary_metric,
        "best_value": best_value,
        "candidates": results,
        "ts_unix": int(time.time()),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = reports_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = reports_dir / "summary.md"
    lines = [
        "# HW4 summary",
        "",
        f"- Best model: `{best_model}`",
        f"- {primary_metric}: `{best_value:.6f}`",
        "",
        "## Candidates",
        "",
    ]
    for row in sorted(results, key=lambda r: float(r[primary_metric]), reverse=True):
        lines.append(f"- `{row['model']}`: {primary_metric}={float(row[primary_metric]):.6f}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[select_best] best={best_model} {primary_metric}={best_value:.6f}")
    print(f"[select_best] copied: {best_model_path} -> {best_out_path}")
    print(f"[select_best] wrote: {summary_json}")


if __name__ == "__main__":
    main()
