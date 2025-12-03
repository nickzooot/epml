# Отчет по настройке проекта Iris MLOps

## Общее описание
- Проект: классификация/регрессия на датасете Iris с полным MLOps-циклом.
- Шаблон и структура созданы через Copier; каталог `src/iris_mlops` содержит код, данные и модели вынесены в отдельные папки (`data`, `models`).

## Управление зависимостями
- Менеджер: **pixi**, конфиг `pixi.toml`.
- Pinned зависимости: Python 3.12, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, pyyaml 6.0.2.
- Dev-инструменты: pre-commit, black, isort, ruff, mypy, bandit.
- Команды:
  - Установка окружения: `pixi install`
  - Активация: `pixi shell` (или префикс `pixi run ...`)
  - Быстрые задачи: `pixi run fmt`, `pixi run lint`, `pixi run types`, `pixi run security`, `pixi run qa`.

## Качество кода и линтеры
- Конфигурация в `pyproject.toml` (Black, isort, Ruff, mypy) и `bandit.yaml` (security).
- Pre-commit: `.pre-commit-config.yaml` с хуками Black, isort, Ruff, mypy, Bandit.
- Основные команды:
  - Форматирование: `pixi run fmt`
  - Линт: `pixi run lint`
  - Типы: `pixi run types`
  - Security: `pixi run security`
  - Полный прогон (может дольше идти из-за скачивания хуков): `pixi run qa`

## Git workflow
- Репозиторий инициализирован: базовая ветка `main`, интеграционная `develop`, пример фичевой ветки `feature/iris-baseline`.
- Правила: работа в `feature/*`, интеграция через `develop`, стабилизация на `main`.
- `.gitignore` настроен под ML (данные, модели, кэши инструментов).

## Версионирование данных (DVC)
- Настроен DVC в поддиректории проекта (`.dvc/config`, subdir=true), remote: локальная папка `../dvc-remote`.
- Stage `get_data` в `dvc.yaml`: запускает `python scripts/get_data.py`, сохраняет датасет в `data/raw/iris.csv` (игнорируется git, хранится через DVC; метаданные в `dvc.lock`).
- Использование: `pixi shell`, затем `dvc repro get_data` для генерации, `dvc push` / `dvc pull` для синхронизации с локальным remote.

## Версионирование моделей (MLflow)
- MLflow tracking/registry: `sqlite:///mlruns.db`, артефакты в `mlruns/` (оба gitignored).
- Обучение и регистрация: `pixi run train` (скрипт `scripts/train_model.py`), логирует метрики/параметры, регистрирует модель `iris-classifier`.
- Просмотр версий: `pixi run list-models`; для подробного сравнения — UI `mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root mlruns`.
- Метаданные: теги `stage=baseline`, `data_version=dvc:get_data`, метрики — accuracy; параметры — тип модели, max_iter.

## Docker
- Dockerfile основан на `python:3.12-slim`, устанавливает pinned зависимости и выставляет `PYTHONPATH=/app/src`.
- Билд: `docker build -t iris-mlops .`
- Запуск ( smoke ): `docker run --rm iris-mlops`

## Краткий чек-лист выполнения
- [x] Структура проекта создана через Copier, README заполнен.
- [x] Зависимости зафиксированы в `pyproject.toml` и `pixi.toml`.
- [x] Инструменты качества кода и pre-commit настроены.
- [x] Git workflow описан и ветки созданы (`develop`, `feature/iris-baseline`).
- [x] Dockerfile добавлен для контейнеризации.
- [x] DVC настроен, локальный remote `../dvc-remote`, создан stage `get_data` с данными iris.
- [x] MLflow настроен локально (sqlite + artifacts), скрипты для обучения/регистрации и просмотра версий добавлены.
