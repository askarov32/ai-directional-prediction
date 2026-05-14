# Задача для Codex: data experiments, predict curls и графики для 4 model services

Ты — senior ML engineer и backend/data engineer.

Проект:

https://github.com/askarov32/ai-directional-prediction

В проекте есть 4 model service:

1. `pinn-service`
2. `mgn-service`
3. `fno-service`
4. `transformer-service`

Нужно перейти к data/experiment части проекта: подготовить reproducible pipeline, который прогоняет одинаковые входные данные через все 4 model-service `/predict`, сохраняет результаты и строит графики для сравнения моделей.

---

## Главная цель

Нужно сделать небольшой, понятный и воспроизводимый эксперимент:

- взять несколько десятков входных вариантов;
- использовать одинаковые физические условия для всех моделей;
- прогнать каждый вариант через 4 `/predict` endpoint;
- сохранить ответы моделей;
- построить графики;
- задокументировать 4 `curl`-запроса — по одному для каждого model-service.

Пример:

- материалы: `sandstone`, `basalt`;
- одинаковые температуры;
- одинаковые boundary conditions;
- одинаковые координаты / временные точки;
- 20 вариантов входных данных;
- 4 запроса на каждый вариант;
- всего примерно 80 predict-запросов.

---

## Важно

Не делай хаотичный рефакторинг проекта.

Работай небольшими шагами:

1. сначала найди реальные predict endpoints и JSON contracts;
2. потом задокументируй curl-запросы;
3. потом предложи список подходящих графиков;
4. потом сделай генератор входных данных;
5. потом сделай runner, который вызывает 4 сервиса;
6. потом сделай генератор графиков;
7. потом обнови документацию.

Если где-то контракт отличается между сервисами — не выдумывай. Найди текущие schemas/DTO в коде и сделай adapter/mapping.

---

# Что нужно сделать

## Step 1 — Найти реальные predict endpoints

Изучи:

- `docker-compose.yml`
- `.env.example`
- backend model clients
- API schemas
- `pinn-service`
- `mgn-service`
- `fno-service`
- `transformer-service`
- README каждого model-service

Найди реальные URL, порты и payload format для:

- PINN `/predict`
- MGN `/predict`
- FNO `/predict`
- Transformer `/predict`

Проверь, отличаются ли request/response schemas между сервисами.

---

## Step 2 — Задокументировать 4 curl-запроса

Создай или обнови документ:

```text
docs/MODEL_SERVICE_CURLS.md
```

В нем обязательно должно быть 4 рабочих curl-запроса:

```text
1. curl для pinn-service /predict
2. curl для mgn-service /predict
3. curl для fno-service /predict
4. curl для transformer-service /predict
```

Для каждого curl укажи:

- local URL;
- Docker Compose service URL, если отличается;
- полный JSON payload;
- пример успешного response;
- возможные ошибки;
- как проверить `/health` и `/ready`.

Пример структуры документа:

````md
# Model Service Predict Curls

## 1. PINN service

### Health

```bash
curl http://localhost:<PORT>/health
```

### Ready

```bash
curl http://localhost:<PORT>/ready
```

### Predict

```bash
curl -X POST http://localhost:<PORT>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "...": "..."
  }'
```

## 2. MGN service

...

## 3. FNO service

...

## 4. Transformer service

...
````

Не пиши примерные порты наугад. Возьми их из `docker-compose.yml` и `.env.example`.

---

## Step 3 — Предложить список графиков

Создай документ:

```text
docs/DATA_EXPERIMENT_CHARTS.md
```

В нем напиши список графиков, которые лучше всего подходят для задачи сравнения моделей thermoelastic wave prediction.

Обязательно рассмотри такие графики:

1. **Temperature prediction comparison**
   - сравнение `T` между PINN / MGN / FNO / Transformer.

2. **Displacement components comparison**
   - сравнение `u`, `v`, `w` по моделям.

3. **Displacement magnitude**
   - график `sqrt(u^2 + v^2 + w^2)` для каждой модели.

4. **Material comparison**
   - sandstone vs basalt при одинаковых условиях.

5. **Model disagreement plot**
   - насколько сильно модели расходятся между собой по `T`, `u`, `v`, `w`.

6. **Prediction vs time**
   - изменение предсказаний по времени `t`.

7. **Prediction vs spatial coordinate**
   - изменение по `x`, `y` или `z`, если формат данных позволяет.

8. **Heatmap / grid plot**
   - особенно для FNO, если он возвращает grid-like output.

9. **Radar / summary chart**
   - агрегированное сравнение моделей по средним значениям, max/min, variance.

10. **Error/fallback chart**
   - какие сервисы ответили нормально, какие ушли в fallback или вернули ошибку.

Для каждого графика укажи:

- зачем он нужен;
- какие поля использует;
- какой файл будет генерироваться;
- какой формат вывода: `.png`, `.csv`, `.json`.

---

## Step 4 — Сделать генератор входных данных

Создай скрипт:

```text
scripts/generate_experiment_inputs.py
```

Он должен генерировать примерно 20 вариантов входных данных.

Минимальные условия:

- материалы: `sandstone`, `basalt`;
- одинаковый набор температур;
- одинаковые boundary conditions;
- одинаковые coordinates/time grid;
- reproducible seed;
- результат сохранять в JSON/JSONL.

Пример результата:

```text
artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
```

Каждая строка должна содержать один experiment case:

```json
{
  "case_id": "case_001_sandstone",
  "material": "sandstone",
  "temperature": 300,
  "boundary_conditions": {
    "type": "fixed",
    "description": "same baseline BC for all services"
  },
  "input": {
    "...": "..."
  }
}
```

Если разные сервисы требуют разные payload schemas, сделай общий canonical input format и отдельный mapper/adapter для каждого сервиса.

---

## Step 5 — Сделать runner для 4 predict-запросов

Создай скрипт:

```text
scripts/run_model_service_experiment.py
```

Он должен:

- читать generated inputs;
- для каждого case вызывать 4 сервиса:
  - PINN;
  - MGN;
  - FNO;
  - Transformer;
- сохранять raw request;
- сохранять raw response;
- сохранять normalized response;
- сохранять ошибки;
- не падать полностью, если один сервис вернул ошибку;
- иметь timeout;
- иметь CLI flags.

Пример запуска:

```bash
python scripts/generate_experiment_inputs.py \
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --num-cases 20

python scripts/run_model_service_experiment.py \
  --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --output-dir artifacts/data_experiments/results \
  --timeout-seconds 20
```

Результаты сохранить в:

```text
artifacts/data_experiments/results/raw/
artifacts/data_experiments/results/normalized/
artifacts/data_experiments/results/summary.csv
artifacts/data_experiments/results/summary.json
```

---

## Step 6 — Сделать генератор графиков

Создай скрипт:

```text
scripts/generate_model_comparison_charts.py
```

Он должен читать:

```text
artifacts/data_experiments/results/summary.csv
```

и генерировать графики в:

```text
artifacts/data_experiments/charts/
```

Минимальный набор графиков:

```text
temperature_comparison.png
displacement_components_comparison.png
displacement_magnitude_comparison.png
material_comparison_sandstone_vs_basalt.png
model_disagreement.png
prediction_vs_time.png
service_status_summary.png
```

Если данных для какого-то графика недостаточно — скрипт должен не падать, а написать понятное предупреждение.

---

## Step 7 — Обновить документацию

Создай или обнови:

```text
docs/MODEL_SERVICE_CURLS.md
docs/DATA_EXPERIMENT_CHARTS.md
docs/DATA_EXPERIMENT_PIPELINE.md
```

В `docs/DATA_EXPERIMENT_PIPELINE.md` опиши:

- цель эксперимента;
- какие сервисы сравниваются;
- как генерируются inputs;
- как запускаются 4 predict-запроса;
- куда сохраняются outputs;
- как строятся графики;
- команды запуска;
- troubleshooting.

---

# Требования к реализации

## Нельзя

- Не ломай существующие PINN / MGN / FNO / Transformer.
- Не удаляй mock services без отдельного обоснования.
- Не переписывай backend architecture.
- Не меняй существующие model contracts без необходимости.
- Не вставляй огромные данные прямо в репозиторий.
- Не генерируй тяжелые artifacts в git, если они должны быть ignored.

## Нужно

- Использовать текущие schemas/contracts проекта.
- Работать через Docker Compose-compatible URLs.
- Сделать CLI scripts.
- Сделать outputs reproducible.
- Добавить `.gitignore` правила для generated artifacts, если нужно.
- Добавить минимальные tests, если в проекте уже есть testing structure для scripts.
- Все curl-запросы должны быть проверяемыми и задокументированными.

---

# Ожидаемый результат

После выполнения задачи должны появиться:

```text
docs/MODEL_SERVICE_CURLS.md
docs/DATA_EXPERIMENT_CHARTS.md
docs/DATA_EXPERIMENT_PIPELINE.md

scripts/generate_experiment_inputs.py
scripts/run_model_service_experiment.py
scripts/generate_model_comparison_charts.py

artifacts/data_experiments/
  inputs/
  results/
  charts/
```

Если `artifacts/` не должен попадать в git, добавь соответствующие правила в `.gitignore`.

---

# Формат ответа после выполнения

Не вставляй полный код файлов в ответ.

Ответь только так:

````md
## Changed files

| File | Action | Summary |
|---|---|---|

## What was implemented

Кратко по пунктам.

## How to run

```bash
docker compose up -d pinn-service mgn-service fno-service transformer-service

python scripts/generate_experiment_inputs.py \
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --num-cases 20

python scripts/run_model_service_experiment.py \
  --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --output-dir artifacts/data_experiments/results

python scripts/generate_model_comparison_charts.py \
  --input artifacts/data_experiments/results/summary.csv \
  --output-dir artifacts/data_experiments/charts
```

## Validation

Какие команды запускались и что прошло.

## Remaining issues

Что осталось, если что-то не удалось.
````
