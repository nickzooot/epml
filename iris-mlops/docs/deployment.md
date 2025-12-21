# Deployment guide

## 1) Python env
Использую **pixi** (все зависимости зафиксированы в `pixi.toml`):
```bash
cd iris-mlops
pixi install --locked
```

## 2) DVC / MLflow / W&B
См. `README.md` в корне проекта: `iris-mlops/README.md`.

## 3) Airflow (HW4)
```bash
cd iris-mlops
pixi run airflow-init
pixi run airflow-test-hw4
```

## 4) ClearML (HW5)
```bash
cd iris-mlops/clearml
docker compose --env-file .env.example -f docker-compose.yml up -d
curl -s http://localhost:8008/debug.ping
```

Экспорт env для клиента:
```bash
export CLEARML_API_HOST="http://localhost:8008"
export CLEARML_WEB_HOST="http://localhost:8080"
export CLEARML_FILES_HOST="http://localhost:8081"
export CLEARML_API_ACCESS_KEY="hw5_local_access"
export CLEARML_API_SECRET_KEY="hw5_local_secret"
```

## 5) Docs (MkDocs)
```bash
cd iris-mlops
pixi run docs-build
```
