# Usage examples

## Formatting / QA
```bash
cd iris-mlops
pixi run fmt
pixi run lint
```

## DVC data stage
```bash
cd iris-mlops
pixi run dvc-repro-get-data
```

## HW4 pipeline (Airflow DAG as local test)
```bash
cd iris-mlops
pixi run airflow-test-hw4
```

## HW5 experiments (ClearML)
```bash
cd iris-mlops
pixi run hw5-smoke
pixi run hw5-experiments
pixi run hw5-compare
pixi run hw5-list-models
```

## Build docs locally
```bash
cd iris-mlops
pixi run docs-serve
```
