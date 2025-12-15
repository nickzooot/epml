from __future__ import annotations

import argparse
import os
import pathlib
from typing import Any

import pandas as pd

import wandb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query & compare W&B runs (online API).")
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY"),
        help="W&B entity (team/user)",
    )
    parser.add_argument("--project", default="iris-mlops-hw3", help="W&B project name")
    parser.add_argument("--group", default=None, help="Filter by run group")
    parser.add_argument("--tag", default=None, help="Filter by tag")
    parser.add_argument("--min-accuracy", type=float, default=None, help="Filter by accuracy >= X")
    parser.add_argument("--sort-by", default="accuracy", help="Sort key (default: accuracy)")
    parser.add_argument("--top-k", type=int, default=20, help="Print top-k rows")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.entity:
        raise SystemExit("Missing --entity (or set WANDB_ENTITY).")

    filters: dict[str, Any] = {}
    if args.group:
        filters["group"] = args.group
    if args.tag:
        filters["tags"] = {"$in": [args.tag]}
    if args.min_accuracy is not None:
        filters["summary_metrics.accuracy"] = {"$gte": args.min_accuracy}

    try:
        api = wandb.Api()
        runs = api.runs(f"{args.entity}/{args.project}", filters=filters)
    except Exception as e:
        raise SystemExit(
            "W&B API request failed. Ensure you are authenticated "
            "(set WANDB_API_KEY or run `wandb login`).\n"
            f"Error: {e}"
        ) from e

    rows: list[dict[str, Any]] = []
    for run in runs:
        summary = dict(run.summary) if run.summary else {}
        config = dict(run.config) if run.config else {}
        rows.append(
            {
                "run_id": run.id,
                "name": run.name,
                "group": run.group,
                "state": run.state,
                "created_at": run.created_at,
                "model_name": config.get("model_name"),
                "accuracy": summary.get("accuracy"),
                "f1_macro": summary.get("f1_macro"),
                "precision_macro": summary.get("precision_macro"),
                "recall_macro": summary.get("recall_macro"),
                "runtime_sec": summary.get("runtime_sec"),
                "url": run.url,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False, na_position="last")

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    if df.empty:
        print("No runs found.")
        return

    print(df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
