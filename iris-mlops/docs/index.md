# Iris MLOps — documentation

Этот сайт собирается **автоматически** через MkDocs и содержит:
- документацию по проекту (развертывание, использование),
- отчеты по экспериментам (HW3–HW5),
- авто‑генерируемые сводные таблицы/графики по экспериментам.

## Быстрый старт
1) Установить окружение (pixi):
```bash
cd iris-mlops
pixi install --locked
```

2) Запустить документацию локально (автоматически синхронизирует отчеты + генерирует summary):
```bash
cd iris-mlops
pixi run docs-serve
```

3) Собрать статический сайт:
```bash
cd iris-mlops
pixi run docs-build
```

## GitHub Pages
Публикация на GitHub Pages настроена через workflow `.github/workflows/docs.yml` (нужно включить `Settings → Pages → Source: GitHub Actions`).

## Репорты
Отчеты лежат в `iris-mlops/reports/*.md` и автоматически синхронизируются в `iris-mlops/docs/reports/`.
