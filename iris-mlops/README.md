# Iris MLOps

Classification/regression on the Iris dataset with full MLOps workflow.

## Dataset
- Uses the `iris` dataset (default: Iris) for quick classification/regression experiments.
- Keep raw data under `data/raw/` and processed features under `data/processed/`.

## Project layout
- `src/iris_mlops/` — package entrypoint for data prep, feature engineering, and modeling code.
- `configs/` — configuration stubs for experiments and training runs.
- `data/raw` and `data/processed` — input data and intermediate artifacts (tracked via placeholders).
- `models/` — saved model binaries or checkpoints.
- `notebooks/` — exploratory work; keep them lightweight.
- `reports/` — experiment summaries and plots.
- `scripts/` — CLI utilities (data download, training, evaluation).
- `tests/` — unit and integration tests.

## Getting started
1) Create a virtual environment and install your chosen dependency manager (to be added in later steps).
2) Populate `configs/base.yaml` with experiment settings (seed, paths, model defaults).
3) Add data prep and training code in `src/iris_mlops/`.

## Dependencies & environment (pixi)
- Pinned runtime deps: numpy==1.26.4, pandas==2.2.3, scikit-learn==1.5.2, pyyaml==6.0.2 (Python 3.12). Dev tools (black, isort, ruff, mypy, bandit, pre-commit) are included too.
- Install pixi (https://pixi.sh) and create the env: `pixi install` (uses `pixi.toml`).
- Activate env: `pixi shell` (or prefix commands with `pixi run ...`).
- Handy tasks: `pixi run fmt`, `pixi run lint`, `pixi run types`, `pixi run security`, `pixi run qa` (full pre-commit sweep).

## Data versioning (DVC)
- Config in `.dvc/config` (subdir mode), local remote `../dvc-remote`.
- Stage `get_data` in `dvc.yaml` runs `python scripts/get_data.py` and produces `data/raw/iris.csv` (gitignored, tracked via DVC with `dvc.lock`).
- Commands (inside pixi env): `dvc repro get_data` to refresh data, `dvc push` / `dvc pull` to sync with the local remote.

## Git workflow
- Repo initialized on `main`; integration branch `develop` and sample feature branch `feature/iris-baseline` are created.
- Work in `feature/*`, merge to `develop` after review, and promote stable changes to `main`.
- Quick commands: `git checkout develop`, `git checkout -b feature/your-task`, merge back via PR or `git merge feature/your-task`.
- `.gitignore` tuned for ML (data, models, tool caches excluded).

## Code quality
- Tooling configured via `pyproject.toml` (Black, isort, Ruff, mypy) and `bandit.yaml`.
- Install pre-commit and enable hooks: `pip install pre-commit && pre-commit install`.
- Run checks manually: `pre-commit run --all-files` or individual tools (e.g., `ruff check .`, `black .`, `bandit -c bandit.yaml -r src`).

## Docker
- Build: `docker build -t iris-mlops .`
- Run (example): `docker run --rm iris-mlops`

## Template usage (Copier)
This project was generated from the local Copier template in `template_mlops/`.

To create a new project from the template:
```bash
copier copy --trust --defaults template_mlops your-new-project
```
Customize answers during the prompt or pass `--data key=value` flags to override defaults.
