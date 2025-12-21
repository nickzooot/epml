from __future__ import annotations

import os
import time
import urllib.request

from clearml import Task


def _require_env() -> None:
    required = [
        "CLEARML_API_HOST",
        "CLEARML_WEB_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "ClearML client is not configured. Missing env vars: "
            f"{missing_str}. See `iris-mlops/clearml/.env.example`."
        )


def _ping(api_host: str) -> None:
    url = api_host.rstrip("/") + "/debug.ping"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
        body = resp.read().decode("utf-8", errors="replace").strip()
    if "OK" not in body:
        raise RuntimeError(f"ClearML ping failed: {url} -> {body!r}")


def main() -> None:
    _require_env()
    api_host = os.environ["CLEARML_API_HOST"]
    _ping(api_host)

    task = Task.init(
        project_name="iris-mlops-hw5",
        task_name="smoke-test",
        task_type=Task.TaskTypes.testing,
        tags=["hw5", "smoke"],
        reuse_last_task_id=False,
    )
    logger = task.get_logger()
    logger.report_scalar(title="metrics", series="smoke_ok", value=1.0, iteration=0)
    task.upload_artifact("meta", {"ts_unix": int(time.time()), "api_host": api_host})
    task.close()
    print("[hw5] ClearML smoke-test completed")


if __name__ == "__main__":
    main()
