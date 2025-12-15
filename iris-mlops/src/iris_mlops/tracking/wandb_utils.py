from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import netrc
import os
import pathlib
import time
from functools import lru_cache
from typing import Any, Callable, Iterator, Mapping, ParamSpec, Sequence, TypeVar

import numpy as np
import pandas as pd

import wandb

P = ParamSpec("P")
R = TypeVar("R")

JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


def _to_jsonable(value: Any, *, max_list_len: int = 50) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, pathlib.Path):
        return str(value)

    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape)}

    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "shape": [value.shape[0], value.shape[1]],
            "cols": list(value.columns),
        }

    if isinstance(value, pd.Series):
        return {"type": "series", "shape": [value.shape[0]]}

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if len(items) > max_list_len:
            items = items[:max_list_len] + ["..."]
        return [_to_jsonable(v) for v in items]

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value  # type: ignore[return-value]


def resolve_wandb_mode(mode: str | None = None) -> str:
    if mode:
        return mode
    if os.getenv("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    return "online" if os.getenv("WANDB_API_KEY") else "offline"


def _has_netrc_credentials(host: str = "api.wandb.ai") -> bool:
    try:
        auth = netrc.netrc().authenticators(host)
    except (FileNotFoundError, OSError, netrc.NetrcParseError):
        return False
    return auth is not None


@lru_cache(maxsize=2)
def _ensure_wandb_login_cached(mode: str, api_key: str | None) -> str:
    if mode != "online":
        return mode

    key = api_key or os.getenv("WANDB_API_KEY")
    if key:
        wandb.login(key=key, relogin=True)
        return "online"

    if not _has_netrc_credentials():
        print(
            "WANDB_API_KEY is not set and no stored login found "
            "(run `wandb login`) -> falling back to WANDB_MODE=offline"
        )
        return "offline"

    try:
        ok = wandb.login()
    except Exception:
        ok = False

    if not ok:
        print(
            "W&B login failed and WANDB_API_KEY is not set -> falling back to WANDB_MODE=offline"
        )
        return "offline"

    return "online"


def ensure_wandb_login(*, api_key: str | None = None, mode: str | None = None) -> str:
    resolved_mode = resolve_wandb_mode(mode)
    return _ensure_wandb_login_cached(resolved_mode, api_key)


@contextlib.contextmanager
def wandb_run(
    *,
    project: str,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    job_type: str | None = None,
    tags: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
    mode: str | None = None,
    api_key: str | None = None,
    run_dir: str | None = None,
) -> Iterator[wandb.sdk.wandb_run.Run]:
    resolved_mode = ensure_wandb_login(api_key=api_key, mode=mode)
    safe_config: dict[str, JsonValue] = {}
    if config:
        safe_config = {str(k): _to_jsonable(v) for k, v in config.items()}

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        job_type=job_type,
        tags=list(tags) if tags else None,
        config=safe_config,
        mode=resolved_mode,
        reinit="finish_previous",
        dir=run_dir,
    )
    try:
        yield run
    finally:
        wandb.finish()


def log_params(params: Mapping[str, Any]) -> None:
    if wandb.run is None:
        return
    safe_params = {str(k): _to_jsonable(v) for k, v in params.items()}
    wandb.config.update(safe_params, allow_val_change=True)


def log_metrics(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    if wandb.run is None:
        return
    numeric: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            numeric[str(k)] = float(v)
    if numeric:
        wandb.log(numeric, step=step)


def log_file_artifact(
    *,
    name: str,
    artifact_type: str,
    path: pathlib.Path,
    aliases: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if wandb.run is None:
        return
    artifact = wandb.Artifact(
        name=name,
        type=artifact_type,
        metadata={str(k): _to_jsonable(v) for k, v in (metadata or {}).items()},
    )
    artifact.add_file(str(path))
    wandb.log_artifact(artifact, aliases=list(aliases) if aliases else None)


def wandb_autolog(func: Callable[P, R]) -> Callable[P, R]:
    signature = None
    try:
        import inspect

        signature = inspect.signature(func)
    except Exception:
        signature = None

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()

        if wandb.run is not None and signature is not None:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            log_params(bound.arguments)

        result = func(*args, **kwargs)
        duration_sec = time.perf_counter() - start
        log_metrics({"runtime_sec": duration_sec})

        if isinstance(result, Mapping):
            log_metrics(result)

        return result

    return wrapper
