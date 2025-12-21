from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

import hydra
from omegaconf import DictConfig


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../configs/hw4", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = _project_root()
    reports_dir = project_root / pathlib.Path(str(cfg.paths.reports))
    summary_path = reports_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path} (run select_best first)")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_model = summary.get("best_model")
    primary_metric = summary.get("primary_metric")
    best_value = summary.get("best_value")

    ts = datetime.now(timezone.utc).isoformat()
    message = (
        f"[{ts}] HW4 Airflow pipeline finished: best_model={best_model} "
        f"{primary_metric}={best_value}"
    )
    print(message)

    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "notifications.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")

    print(f"[notify] appended: {log_path}")


if __name__ == "__main__":
    main()
