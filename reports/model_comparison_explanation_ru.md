# Объяснение результатов графиков

Источник данных: `artifacts/data_experiments/results/summary.csv`

Прогон:

- кейсов: `40`
- ответов моделей: `160`
- успешных ответов: `160`
- fallback-ответов: `0`
- ошибок: `0`

## 1. Общий статус сервисов

Графики:

- `service_status_summary.png`
- `model_validity_summary.png`

Все четыре сервиса успешно ответили на все 40 сценариев:

- `PINN`: checkpoint mode
- `MGN`: rollout mode
- `FNO`: checkpoint mode
- `Transformer`: checkpoint mode

Это значит, что текущий набор графиков построен не по mock/fallback-ответам. `MGN` теперь действительно работает через rollout, а `FNO` действительно загрузил checkpoint.

Главное замечание: у `FNO` все 40 ответов помечены как outlier по масштабу `max_displacement` и `max_temperature_perturbation`. Поэтому в scientific valid-only графиках он может отображаться как исключенный, но это уже не fallback.

## 2. Время распространения волны

Графики:

- `basalt_vs_sandstone_travel_time.png`
- `material_comparison_sandstone_vs_basalt.png`
- `prediction_vs_time.png`
- `heatmap_case_model_travel_time.png`
- `heatmap_model_disagreement_travel_time.png`
- `heatmap_material_model_travel_time.png`
- `heatmap_time_model_travel_time.png`
- `heatmap_probe_z_model_travel_time.png`

Среднее предсказанное время распространения:

| Model | Mean travel time, ms | Min | Max |
|---|---:|---:|---:|
| PINN | 0.109 | 0.079 | 0.129 |
| Transformer | 0.137 | 0.100 | 0.159 |
| FNO | 1.294 | 0.746 | 1.753 |
| MGN | 2.467 | 1.384 | 3.606 |

Интерпретация:

- `PINN` и `Transformer` дают самые быстрые времена прохождения.
- `FNO` дает промежуточные времена, но его надо читать осторожно, потому что он работает через `rect_3d_to_rect_2d`.
- `MGN` дает самые большие времена, потому что его rollout-ответ заметно отличается по масштабу от PINN/Transformer.
- Для всех моделей sandstone в среднем получается чуть медленнее basalt.

По материалам:

| Material | PINN | MGN | FNO | Transformer |
|---|---:|---:|---:|---:|
| basalt | 0.107 | 2.287 | 1.238 | 0.133 |
| sandstone | 0.112 | 2.711 | 1.370 | 0.143 |

Вывод: графики времени показывают, что материал влияет на результат, но сильнее всего различаются сами модельные семейства.

## 3. Смещение

Графики:

- `displacement_magnitude_comparison.png`
- `max_displacement_valid_only.png`
- `max_displacement_log_diagnostic.png`
- `basalt_vs_sandstone_displacement.png`
- `heatmap_case_model_displacement.png`
- `heatmap_model_disagreement_displacement.png`
- `heatmap_material_model_displacement.png`
- `heatmap_pressure_model_displacement.png`

Среднее максимальное смещение:

| Model | Mean max displacement |
|---|---:|
| PINN | 1.46e-05 |
| Transformer | 1.48e-03 |
| MGN | 2.05e-03 |
| FNO | 1.78e+07 |

Интерпретация:

- `PINN` дает самое маленькое смещение.
- `Transformer` и `MGN` находятся в близком порядке величины: примерно `1e-3`.
- `MGN` сейчас почти константный по `max_displacement`: `0.0020458` во всех кейсах. Это значит, что rollout сервис возвращает стабильную summary-метрику, но чувствительность этой метрики к входам пока слабая.
- `FNO` дает физически нереалистичный масштаб: примерно `17 790 000`. Поэтому он исключается из `valid-only` графика и показывается в log diagnostic графике.

Вывод: для обычного физического сравнения displacement сейчас лучше смотреть `PINN`, `MGN`, `Transformer`. `FNO` в этом показателе пока является диагностическим сигналом проблемы нормализации или denormalization.

## 4. Температурное возмущение

Графики:

- `temperature_comparison.png`
- `temperature_perturbation_valid_only.png`
- `temperature_perturbation_log_diagnostic.png`
- `heatmap_case_model_temperature.png`
- `heatmap_model_disagreement_temperature.png`
- `heatmap_material_model_temperature.png`
- `heatmap_temperature_model_temperature_perturbation.png`

Среднее максимальное температурное возмущение:

| Model | Mean max temperature perturbation |
|---|---:|
| PINN | 0.154 |
| MGN | 3.211 |
| Transformer | 526.154 |
| FNO | 2.47e+06 |

Интерпретация:

- `PINN` дает самый спокойный температурный отклик.
- `MGN` возвращает стабильное значение около `3.21`.
- `Transformer` дает значительно более сильный температурный отклик. При этом для sandstone он выше, чем для basalt.
- `FNO` снова находится в экстремальном масштабе и помечается как outlier.

По материалам:

| Material | PINN | MGN | FNO | Transformer |
|---|---:|---:|---:|---:|
| basalt | 0.159 | 3.211 | 2.47e+06 | 372.822 |
| sandstone | 0.147 | 3.211 | 2.47e+06 | 733.604 |

Вывод: temperature charts показывают большой разрыв между моделями. Особенно заметно, что `Transformer` чувствительнее к материалу, а `FNO` требует отдельной проверки масштаба.

## 5. Направление волны

Графики:

- `displacement_components_comparison.png`
- `azimuth_circular_disagreement_by_case.png`
- `elevation_comparison.png`
- `heatmap_model_disagreement_azimuth.png`
- `heatmap_model_disagreement_elevation.png`

Средние углы:

| Model | Mean azimuth, deg | Mean elevation, deg |
|---|---:|---:|
| PINN | 11.485 | 45.533 |
| MGN | 12.439 | 37.965 |
| FNO | -164.706 | 0.000 |
| Transformer | 13.484 | 59.459 |

Интерпретация:

- `PINN`, `MGN`, `Transformer` дают похожий азимут: примерно `11-13` градусов.
- `Transformer` дает самый высокий elevation, то есть сильнее направляет волну вверх/вглубь по 3D-компоненте.
- `MGN` дает самый умеренный elevation среди 3D-моделей.
- `FNO` имеет `direction_z = 0` и `elevation = 0`, потому что текущий `FNO` работает как `FNO2d` и получает адаптированный домен `rect_3d_to_rect_2d`.
- Азимут `FNO` около `-164.7` градусов сильно отличается от остальных, поэтому angular disagreement heatmap показывает его как отдельный проблемный кластер.

Вывод: для 3D directional prediction сейчас корректнее сравнивать `PINN`, `MGN`, `Transformer`. `FNO` пока нельзя считать полноценным 3D-направленным предиктором.

## 6. Глубина и 3D-чувствительность

Графики:

- `depth_sensitivity.png`
- `depth_sensitivity_travel_time.png`
- `depth_sensitivity_displacement.png`
- `depth_sensitivity_temperature.png`
- `domain_adaptation_summary.png`

По `probe_z` видно:

- `PINN` меняет travel time умеренно: примерно от `0.082` до `0.121 ms`.
- `MGN` сильнее реагирует на глубину: примерно от `1.996` до `3.096 ms`.
- `Transformer` тоже меняет travel time: примерно от `0.103` до `0.154 ms`.
- `FNO` имеет численную зависимость по travel time, но физически это не полноценная 3D-зависимость, потому что домен был сжат до 2D.

`domain_adaptation_summary.png` показывает важный факт:

- `PINN`, `MGN`, `Transformer`: `rect_3d -> rect_3d`
- `FNO`: `rect_3d -> rect_2d`

Вывод: 3D-сценарий уже работает для трех моделей, но `FNO` пока остается 2D-веткой.

## 7. Heatmaps

Графики:

- `heatmap_case_model_*`
- `heatmap_model_disagreement_*`
- `heatmap_material_model_*`
- `heatmap_time_model_*`
- `heatmap_probe_z_model_*`
- `heatmap_temperature_model_*`
- `heatmap_pressure_model_*`

Что показывают heatmaps:

- `case x model` heatmaps позволяют быстро увидеть, где конкретная модель выбивается по отдельному кейсу.
- `model disagreement` heatmaps показывают среднее расхождение между парами моделей.
- `material x model` heatmaps показывают, как basalt и sandstone меняют средний ответ.
- parameter heatmaps показывают, как метрики меняются при разных `time_ms`, `probe_z`, `temperature_c`, `pressure_mpa`.

Главный вывод из heatmaps:

- `PINN` и `Transformer` часто ближе друг к другу по travel time.
- `MGN` систематически дает более высокое travel time.
- `FNO` резко отличается по displacement/temperature из-за масштаба и по direction из-за 2D-адаптации.

## 8. Итоговая интерпретация

Текущая демонстрация уже хороша как comparison pipeline:

- все сервисы отвечают без fallback;
- backend/analytics корректно различают checkpoint, rollout и adaptation;
- 3D-сценарии реально проходят через `PINN`, `MGN`, `Transformer`;
- графики честно показывают ограничения `FNO`.

Научно самая сильная текущая тройка для 3D:

- `PINN`
- `MGN`
- `Transformer`

`FNO` сейчас интеграционно работает, но научно ограничен:

- он 2D-only;
- `direction_z = 0`;
- `elevation = 0`;
- displacement и temperature имеют outlier-масштаб;
- его нельзя использовать как полноценную 3D-модель до исправления `FNO3d` или нормализации/denormalization.

## 9. Короткий текст для презентации

В эксперименте было сформировано 40 трехмерных сценариев для basalt и sandstone. Каждый сценарий был отправлен в четыре модели: PINN, MeshGraphNet, FNO и Transformer. Все 160 запросов завершились успешно, fallback-ответов и ошибок не было.

PINN и Transformer показали самые быстрые времена распространения волны, MeshGraphNet дал более медленный, но стабильный rollout-ответ. Для sandstone времена распространения в среднем выше, чем для basalt, что соответствует ожиданию различий между геологическими средами.

По направлению волны PINN, MeshGraphNet и Transformer формируют полноценные 3D-векторы с ненулевым elevation. Transformer дает самый высокий elevation, MeshGraphNet самый умеренный, PINN находится между ними. FNO в текущей реализации остается 2D-моделью, поэтому его `direction_z` и `elevation` равны нулю.

По displacement и temperature видно, что FNO имеет некорректный масштаб выходов. Поэтому его значения отмечены как outliers и вынесены в диагностические графики. Для полноценного научного сравнения FNO нужно доработать: либо реализовать FNO3d, либо исправить нормализацию и масштабирование выходов.
