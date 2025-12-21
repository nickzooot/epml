from __future__ import annotations

import argparse
from typing import Any

from clearml import Model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List registered ClearML models for HW5")
    parser.add_argument("--project", default="iris-mlops-hw5")
    parser.add_argument("--name", default="iris-hw5-classifier")
    parser.add_argument("--max-results", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    models = Model.query_models(
        project_name=args.project,
        model_name=args.name,
        tags=["hw5", "iris"],
        include_archived=False,
        max_results=args.max_results,
    )
    if not models:
        print("[hw5] No models found. Run `scripts/hw5_clearml_train.py` first.")
        return

    for idx, m in enumerate(models, start=1):
        meta: dict[str, Any] = {}
        try:
            meta = m.get_all_metadata_casted()
        except Exception:
            meta = {}

        print(f"[{idx:02d}] id={m.id} name={m.name} project={m.project}")
        print(f"     tags={m.tags} framework={m.framework}")
        if meta:
            acc = meta.get("metrics_json") or meta.get("accuracy")
            print(f"     metadata.keys={list(meta.keys())} metrics={acc}")
        print(f"     uri={m.url}")


if __name__ == "__main__":
    main()
