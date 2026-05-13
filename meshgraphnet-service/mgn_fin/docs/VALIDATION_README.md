# Валидация MeshGraphNet на COMSOL ground truth

Этот патч добавляет полноценную проверку модели после обучения.

## Что проверяется

1. **One-step validation**  
   Проверяет локальный переход `state_t -> state_{t+1}` на test split.

2. **Autoregressive rollout validation**  
   Модель стартует с первого test-состояния и дальше использует свои собственные предсказания. Это главный тест стабильности.

3. **Derived physical metrics**  
   Считаются ошибки не только по сырым полям, но и по физически понятным величинам:
   - `temperature_change`
   - `displacement_magnitude`
   - `velocity_magnitude`
   - `von_mises_stress`
   - `strain_magnitude`
   - `wave_front_radius`

4. **COMSOL vs Prediction visual comparison**  
   Строятся срезы: COMSOL / MeshGraphNet / absolute error.

## Как запустить

Из корня проекта:

```powershell
python scripts/validate_model.py --config configs/base.yaml --dataset_id sandstone_comsol_real --checkpoint outputs/checkpoints/best_model.pt --split test --slice_axis z
```

Если по `z` плохо видно:

```powershell
python scripts/validate_model.py --config configs/base.yaml --dataset_id sandstone_comsol_real --checkpoint outputs/checkpoints/best_model.pt --split test --slice_axis x
```

или:

```powershell
python scripts/validate_model.py --config configs/base.yaml --dataset_id sandstone_comsol_real --checkpoint outputs/checkpoints/best_model.pt --split test --slice_axis y
```

Для быстрой проверки можно ограничить rollout:

```powershell
python scripts/validate_model.py --config configs/base.yaml --dataset_id sandstone_comsol_real --checkpoint outputs/checkpoints/best_model.pt --split test --max_rollout_steps 5
```

## Что откроется после запуска

```powershell
start outputs\validation\validation_report.html
```

## Главные файлы

```text
outputs/validation/validation_report.html
outputs/validation/validation_summary.json
outputs/validation/tables/metrics_per_field.csv
outputs/validation/tables/group_metrics.csv
outputs/validation/tables/error_over_time.csv
outputs/validation/tables/derived_error_over_time.csv
outputs/validation/tables/physical_summary.csv
outputs/validation/tables/wave_front_radius.csv
outputs/validation/figures/rollout_rmse_selected_fields.png
outputs/validation/figures/derived_rmse_over_time.png
outputs/validation/figures/wave_front_radius.png
outputs/validation/figures/comparison_temperature_change_step*.png
outputs/validation/figures/comparison_displacement_magnitude_step*.png
outputs/validation/figures/comparison_von_mises_stress_step*.png
```

## Как читать

- Если `one_step` ошибка маленькая, а `rollout` быстро растёт — модель локально обучилась, но прогноз нестабилен.
- Если ошибка растёт только по `stress/strain`, но температура нормальная — тепловая часть усвоена лучше механической.
- Если `wave_front_radius` prediction сильно отстаёт/убегает от COMSOL — модель неправильно воспроизводит скорость распространения фронта.
- Для обычной демонстрации показывай `temperature_change`, `displacement_magnitude`, `velocity_magnitude`, `von_mises_stress`.
- Для исследовательского анализа смотри `metrics_per_field.csv` и `error_over_time.csv`.
