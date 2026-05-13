# DATA_FORMAT

Обязательные реальные COMSOL CSV в `datasets/<dataset_id>/raw/`:

- `data_displacement.csv`
- `data_temperature.csv`
- `data_strain.csv`
- `data_stress_1.csv`
- `data_stress_2.csv`
- `data_stress_3.csv`
- `data_materials.csv`

Парсер ищет строку заголовка с `X, Y, Z`, даже если строка начинается с `%`. Динамические колонки распознаются по шаблону:

```text
field_name (unit) @ t=value
```

Извлекаются `field_name`, `unit`, `time`, после чего собирается тензор `trajectories[T, N, F]`.

Если CSV имеют разные координаты или число узлов, поля выравниваются на reference-coordinates через nearest neighbor. Материалы не являются target-полями: они попадают в `node_static`.

Не-COMSOL CSV, например `basalt_mesh.csv`, игнорируются.
