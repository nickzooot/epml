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
