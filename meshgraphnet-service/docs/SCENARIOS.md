# SCENARIOS

Поддерживаемые сценарии задаются в `datasets/<dataset_id>/scenario.yaml`.

## heated_rod

```yaml
scenario:
  type: heated_rod
  initial_temperature: 773.15
  background_temperature: 293.15
  source_center: [0.0, 0.0, 0.0]
  source_radius: 0.01
```

## impact

```yaml
scenario:
  type: impact
  impact_location: [0.5, 0.5, 1.0]
  impact_radius: 0.05
  impact_force: 1000000.0
  impact_duration: 0.0001
  impact_direction: [0.0, 0.0, -1.0]
```

## side_pressure

```yaml
scenario:
  type: side_pressure
  pressure_side: x_min
  pressure_value: 500000.0
  pressure_duration: 0.005
  loading_profile: ramp
```

## building_load

```yaml
scenario:
  type: building_load
  load_area_center: [0.5, 0.5, 1.0]
  load_area_size: [0.3, 0.3]
  load_value: 2000000.0
  load_type: static
  foundation_geometry: rectangular
  duration: 0.01
```
