from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603


def main() -> None:
    base = [sys.executable, "scripts/hw5_clearml_train.py", "clearml.group=hw5-exp"]

    plan: list[list[str]] = [
        ["model=logreg", "model.params.C=0.1"],
        ["model=logreg", "model.params.C=1.0"],
        ["model=logreg", "model.params.C=10.0"],
        ["model=rf", "model.params.n_estimators=100"],
        ["model=rf", "model.params.n_estimators=300"],
        ["model=rf", "model.params.n_estimators=500"],
        ["model=svc", "model.params.C=0.5"],
        ["model=svc", "model.params.C=3.0"],
        ["model=svc", "model.params.C=10.0"],
    ]

    for idx, overrides in enumerate(plan, start=1):
        print(f"\n[{idx:02d}/{len(plan):02d}] overrides: {overrides}")
        _run(base + overrides)

    print("\n[hw5] Done. Use `python scripts/hw5_clearml_compare.py` to list top runs.")


if __name__ == "__main__":
    main()
