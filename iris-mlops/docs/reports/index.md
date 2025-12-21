# Reports

Отчеты по предыдущим заданиям живут в `iris-mlops/reports/*.md`.

Перед сборкой документации они автоматически копируются в `docs/reports/` (и генерируется summary) при запуске:
```bash
cd iris-mlops
pixi run docs-build
```

Дополнительно генерируется страница со сводными таблицами и графиками:
- `docs/reports/experiments_generated.md`

Отчет по HW6 (настройка документации/публикации):
- `reports/hw6_docs.md`
