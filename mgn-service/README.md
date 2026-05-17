# MGN_project: COMSOL → Conditional MeshGraphNet → validation → wave visualization

Это рабочее исследовательское ядро для AI-прогнозирования термоупругих и механических волн в геологических средах. Проект рассчитан на реальные COMSOL-выгрузки, а не на synthetic pipeline.

## 0. Интеграция с основным MVP

В основном `docker-compose.yml` этот сервис подключен как route для модели `meshgraphnet`.

По умолчанию backend ходит сюда:

```text
http://mgn-service:9000/predict
```

Локально сервис открыт на:

```text
http://localhost:9001
```

Сервис поддерживает:

```text
GET /health
GET /ready
POST /predict
```

Если реальные `datasets/` и `outputs/checkpoints/best_model.pt` еще не подготовлены, `MGN_ALLOW_FALLBACK=true` позволяет сервису вернуть валидный demo-response для frontend/backend. Когда dataset и checkpoint появятся, `/predict` запускает реальный `scripts/run_prediction.py`.

Основные переменные:

```bash
MGN_DATASET_ID=sandstone_comsol_real
MGN_CONFIG_PATH=configs/inference.yaml
MGN_CHECKPOINT_PATH=outputs/checkpoints/best_model.pt
MGN_DEVICE=cuda
MGN_PREDICT_TIMEOUT_SECONDS=600
MGN_ROLLOUT_STEPS=5
MGN_ALLOW_FALLBACK=true
```

## 1. Формат нового датасета

Каждый датасет кладётся так:

```text
datasets/<dataset_id>/
├── raw/
│   ├── data_displacement.csv
│   ├── data_temperature.csv
│   ├── data_strain.csv
│   ├── data_stress_1.csv
│   ├── data_stress_2.csv
│   ├── data_stress_3.csv
│   ├── data_materials.csv
│   └── <mesh>.mphtxt          # optional, если нет — будет kNN fallback
├── processed/
└── scenario.yaml
```

Парсер намеренно читает только CSV с префиксами `data_displacement`, `data_temperature`, `data_strain`, `data_stress`, `data_materials`. Файлы вроде `basalt_mesh.csv` игнорируются и не парсятся как физические поля.

## 2. scenario.yaml

Поддерживаются сценарии:

- `heated_rod` — раскалённый прут / тепловой источник;
- `impact` — удар / импульсная механическая нагрузка;
- `side_pressure` — боковое давление;
- `building_load` — нагрузка здания/фундамента.

Шаблоны лежат в `configs/scenarios/`. Для нового датасета скопируй подходящий YAML в `datasets/<dataset_id>/scenario.yaml` и исправь `dataset_id`, `rock_type`, `geometry.mesh_file`, параметры сценария и время.

## 3. Подготовка датасета

```bash
python scripts/prepare_dataset.py --config configs/base.yaml --dataset_id sandstone_comsol_real
python scripts/prepare_dataset.py --config configs/base.yaml --dataset_id basalt_comsol_real
```

Результат:

```text
datasets/<dataset_id>/processed/
├── graph.pt
├── trajectories.pt
├── metadata.json
├── normalization.json
├── dynamic_normalization.json
├── static_normalization.json
└── preview.csv
```

Если `.mphtxt` отсутствует или не совпадает по числу узлов с CSV, проект не падает: строится kNN-граф по координатам из CSV, а в `metadata.json` пишется `graph_source`.

## 3b. Strict 2D bridge из `rod_experiments_2d`

Если основной thesis pipeline уже собрал:

```text
pinn-service/artifacts/rod_experiments_2d/
```

можно не возвращаться к raw CSV для первого 2D-retrain MeshGraphNet. Вместо этого используй bridge-скрипт:

```bash
python scripts/build_2d_mgn_datasets.py \
  --input-root ../pinn-service/artifacts/rod_experiments_2d \
  --registry-root datasets \
  --k-nearest 12 \
  --target-mode delta
```

Он создаёт dataset ids:

```text
datasets/granite_rod_2d
datasets/limestone_rod_2d
datasets/sandstone_rod_2d
datasets/basalt_rod_2d
```

Каждый из них уже содержит:

```text
scenario.yaml
processed/graph.pt
processed/trajectories.pt
processed/metadata.json
processed/normalization.json
```

Для отдельного 2D checkpoint path используй:

```text
configs/train_2d.yaml
```


## 3c. Универсальное переформатирование под MeshGraphNet/FNO/PINN/Transformer

Если нужен датасет, который подходит не только под MeshGraphNet, используй новый скрипт:

```bash
python scripts/reformat_dataset.py \
  --config configs/base.yaml \
  --dataset_id sandstone_comsol_real \
  --formats canonical graph pinn transformer \
  --k_nearest 12
```

Он создаёт:

```text
datasets/<dataset_id>/processed/
├── canonical/      # общий источник истины: coords/time/dynamic/static/masks
├── graph/          # MeshGraphNet adapter
├── fno/            # регулярный grid, если указан --formats fno
├── pinn/           # index для sampling из canonical
├── transformer/    # index для token sampling из canonical
├── graph.pt        # legacy copy для текущего MeshGraphNet training
└── trajectories.pt # legacy copy для текущего MeshGraphNet training
```

Для FNO не включай полный 3D-grid без необходимости: он может занимать гигабайты. Начни так:

```bash
python scripts/reformat_dataset.py \
  --config configs/base.yaml \
  --dataset_id sandstone_comsol_real \
  --formats fno \
  --grid_res 32 32 32 \
  --fno_max_timesteps 128
```

Подробно: `docs/UNIVERSAL_DATASET.md`.

## 4. Обучение базовой модели

```bash
python scripts/train_base_model.py --config configs/base.yaml --dataset_ids sandstone_comsol_real
```

Артефакты:

```text
outputs/checkpoints/best_model.pt
outputs/checkpoints/last_model.pt
outputs/logs/train_history.json
outputs/logs/test_metrics.json
```

Для быстрой проверки:

```bash
python scripts/train_base_model.py --config configs/base.yaml --dataset_ids sandstone_comsol_real --epochs 3
```

Strict 2D baseline:

```bash
python scripts/train_base_model.py --config configs/train_2d.yaml --dataset_ids limestone_rod_2d --epochs 200
```

Multi-rock strict 2D baseline:

```bash
python scripts/train_base_model.py --config configs/train_2d.yaml --dataset_ids granite_rod_2d limestone_rod_2d sandstone_rod_2d basalt_rod_2d --epochs 200
```

## 5. Дообучение на новой породе / сценарии

```bash
python scripts/finetune_model.py \
  --config configs/finetune.yaml \
  --dataset_id basalt_comsol_real \
  --checkpoint outputs/checkpoints/best_model.pt \
  --mode full
```

Режимы:

- `full` — обучаются все слои;
- `decoder_only` — заморожены encoder + processor, обучается decoder;
- `processor_decoder` — заморожен encoder, обучаются processor + decoder.

Артефакты:

```text
outputs/checkpoints_finetuned/best_model.pt
outputs/checkpoints_finetuned/last_model.pt
outputs/logs_finetuned/train_history.json
outputs/logs_finetuned/test_metrics.json
```

## 6. Валидация COMSOL vs MGN

```bash
python scripts/validate_model.py \
  --config configs/base.yaml \
  --dataset_id basalt_comsol_real \
  --checkpoint outputs/checkpoints_finetuned/best_model.pt \
  --split test \
  --slice_axis z
```

Открывать:

```text
outputs/validation/validation_report.html
outputs/validation/tables/metrics_per_field.csv
outputs/validation/tables/error_over_time.csv
outputs/validation/tables/derived_error_over_time.csv
outputs/validation/tables/wave_front_radius.csv
outputs/validation/figures/rollout_rmse_selected_fields.png
outputs/validation/figures/derived_rmse_over_time.png
outputs/validation/figures/wave_front_radius.png
```

## 7. Prediction / rollout

```bash
python scripts/run_prediction.py \
  --config configs/inference.yaml \
  --dataset_id basalt_comsol_real \
  --checkpoint outputs/checkpoints_finetuned/best_model.pt
```

Результат:

```text
outputs/predictions/prediction.pt
outputs/predictions/prediction_nodes.csv
outputs/predictions/summary_metrics.json
```

## 8. Визуализация волн и стрелок

```bash
python scripts/visualize_results.py \
  --dataset_id basalt_comsol_real \
  --prediction outputs/predictions/prediction.pt \
  --slice_axis z
```

Открывать:

```text
outputs/wave_arrows/wave_arrows_report.html
outputs/wave_arrows/animations/quiver_temperature_change_velocity.gif
```

Стрелки строятся не на каждом узле, а с ограничением `--max_arrows`, например 150–300. Фон: `temperature_change`, `von_mises_stress`, `risk_flag`. Векторы: `velocity` (`ut`, `vt`, `wt`) или `displacement` (`u`, `v`, `w`).

## 9. Один запуск с нуля

```bash
python scripts/run_from_scratch.py --dataset_id basalt_comsol_real --epochs 3 --clean --slice_axis z
```

Если передан checkpoint, будет fine-tune:

```bash
python scripts/run_from_scratch.py \
  --dataset_id basalt_comsol_real \
  --checkpoint outputs/checkpoints/best_model.pt \
  --epochs 3 \
  --clean \
  --slice_axis z \
  --mode full
```

`--clean` удаляет `processed/` и generated outputs, но никогда не удаляет `raw/`.

## 10. Если нет .mphtxt

Оставь `geometry.mesh_file` пустым или не указывай его. Подготовка датасета выведет warning и построит kNN-граф по координатам CSV:

```bash
python scripts/prepare_dataset.py --dataset_id basalt_comsol_real --k_nearest 12
```

Для более плотного графа можно поднять `--k_nearest 16` или `24`, но это увеличит память и время обучения.
