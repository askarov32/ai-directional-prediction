# Data Experiment Charts

This document proposes the first chart set for comparing direct predictions from:

- `pinn-service`
- `mgn-service`
- `fno-service`
- `transformer-service`

The goal is not only to compare values, but also to expose:

- model agreement or disagreement
- fallback behavior
- sensitivity to temperature, pressure, frequency, time, and geometry
- differences between geological media under the same forcing

## Normalized fields required first

Before charting, the runner should normalize every direct service response into one canonical record with at least:

```json
{
  "case_id": "case_001",
  "model": "fno",
  "service_mode": "checkpoint",
  "fallback_used": false,
  "material": "sandstone",
  "temperature_c": 120.0,
  "pressure_mpa": 35.0,
  "time_ms": 12.0,
  "frequency_hz": 50.0,
  "source_x": 0.15,
  "source_y": 0.40,
  "source_z": 0.0,
  "probe_x": 0.70,
  "probe_y": 0.55,
  "probe_z": 0.0,
  "direction_x": 0.981194,
  "direction_y": 0.193022,
  "direction_z": 0.0,
  "azimuth_deg": 11.1292,
  "elevation_deg": 0.0,
  "magnitude": 1.0,
  "travel_time_ms_pred": 2.363967,
  "max_displacement": 0.00145952,
  "max_temperature_perturbation": 1.2,
  "wave_type": "fno_skeleton_fallback",
  "status": "ok"
}
```

This normalized table is the basis for all plots below.

## 1. Temperature prediction comparison

Purpose:

- Compare thermal response intensity between the four model services.
- See whether one model systematically predicts stronger or weaker temperature perturbation.

Fields used:

- `model`
- `material`
- `temperature_c`
- `max_temperature_perturbation`

Recommended plot:

- grouped bar chart
- optionally line plot by `temperature_c`

Output files:

- `artifacts/data_experiments/charts/temperature_prediction_comparison.png`
- `artifacts/data_experiments/tables/temperature_prediction_comparison.csv`

## 2. Displacement components comparison

Purpose:

- Compare predicted propagation direction components directly.
- Useful because some models may have similar magnitude but different orientation.

Fields used:

- `direction_x`
- `direction_y`
- `direction_z`
- `model`
- `material`
- `case_id`

Recommended plot:

- grouped bars per case
- one chart each for `x`, `y`, `z`

Output files:

- `artifacts/data_experiments/charts/direction_components_x.png`
- `artifacts/data_experiments/charts/direction_components_y.png`
- `artifacts/data_experiments/charts/direction_components_z.png`
- `artifacts/data_experiments/tables/direction_components.csv`

## 3. Displacement magnitude comparison

Purpose:

- Compare response strength independently from direction.
- Good first summary metric for relative wave-response intensity.

Fields used:

- `magnitude`
- `max_displacement`
- `model`
- `material`

Recommended plot:

- grouped bar chart
- scatter plot `magnitude` vs `max_displacement`

Output files:

- `artifacts/data_experiments/charts/displacement_magnitude_comparison.png`
- `artifacts/data_experiments/charts/magnitude_vs_max_displacement.png`

## 4. Material comparison

Purpose:

- Compare how the same forcing behaves in `sandstone` vs `basalt`.
- Later extend to `granite` and `limestone`.

Fields used:

- `material`
- `model`
- `azimuth_deg`
- `travel_time_ms_pred`
- `max_displacement`
- `max_temperature_perturbation`

Recommended plot:

- small multiples, one panel per metric

Output files:

- `artifacts/data_experiments/charts/material_comparison_dashboard.png`
- `artifacts/data_experiments/tables/material_comparison_summary.csv`

## 5. Model disagreement plot

Purpose:

- Measure how far model outputs are from each other for the same case.
- This is one of the most important comparison charts.

Fields used:

- normalized outputs from all models for the same `case_id`
- disagreement metrics:
  - `std(azimuth_deg)`
  - `std(travel_time_ms_pred)`
  - `std(max_displacement)`
  - `std(max_temperature_perturbation)`
  - pairwise cosine disagreement for direction vectors

Recommended plot:

- heatmap by `case_id x metric`
- pairwise model disagreement matrix

Output files:

- `artifacts/data_experiments/charts/model_disagreement_heatmap.png`
- `artifacts/data_experiments/charts/model_pairwise_disagreement.png`
- `artifacts/data_experiments/tables/model_disagreement_metrics.csv`

## 6. Prediction vs time

Purpose:

- Evaluate sensitivity to `scenario.time_ms`.
- Good for seeing temporal consistency across models.

Fields used:

- `time_ms`
- `azimuth_deg`
- `travel_time_ms_pred`
- `max_displacement`
- `max_temperature_perturbation`
- `model`
- `material`

Recommended plot:

- line plot per model

Output files:

- `artifacts/data_experiments/charts/prediction_vs_time_travel_time.png`
- `artifacts/data_experiments/charts/prediction_vs_time_displacement.png`
- `artifacts/data_experiments/charts/prediction_vs_time_temperature.png`

## 7. Prediction vs spatial coordinate

Purpose:

- Show sensitivity to probe position or source-probe geometry.
- Useful especially for direction changes and travel-time changes.

Fields used:

- `probe_x`
- `probe_y`
- `probe_z`
- `source_x`
- `source_y`
- `source_z`
- `azimuth_deg`
- `travel_time_ms_pred`
- `max_displacement`

Recommended plot:

- line plot if varying only one coordinate
- scatter plot if varying 2D positions

Output files:

- `artifacts/data_experiments/charts/prediction_vs_probe_x.png`
- `artifacts/data_experiments/charts/prediction_vs_probe_y.png`
- `artifacts/data_experiments/charts/source_probe_geometry_scatter.png`

## 8. Heatmap or grid plot

Purpose:

- Best suited for grid-native or field-like outputs, especially `FNO`.
- Can still be used for cross-model summaries if only scalar metrics are available.

Fields used:

- `temperature_c`
- `pressure_mpa`
- `travel_time_ms_pred`
- `max_displacement`
- `max_temperature_perturbation`

Recommended plot:

- heatmap for `temperature x pressure`
- one heatmap per model

Output files:

- `artifacts/data_experiments/charts/temperature_pressure_travel_time_heatmap.png`
- `artifacts/data_experiments/charts/temperature_pressure_displacement_heatmap.png`

## 9. Radar or summary chart

Purpose:

- High-level comparison of model behavior in one figure.
- Good for presentation and thesis demos.

Fields used:

- aggregated means or medians:
  - `azimuth_deg`
  - `travel_time_ms_pred`
  - `magnitude`
  - `max_displacement`
  - `max_temperature_perturbation`
  - fallback rate

Recommended plot:

- radar chart
- or compact summary bar panel if radar looks too noisy

Output files:

- `artifacts/data_experiments/charts/model_summary_radar.png`
- `artifacts/data_experiments/tables/model_summary_metrics.csv`

## 10. Error and fallback chart

Purpose:

- Show operational robustness of the services.
- Important because some services may answer in fallback mode or fail readiness.

Fields used:

- `status`
- `fallback_used`
- `service_mode`
- `model`
- `case_id`
- optional error code and message

Recommended plot:

- stacked bar chart by model
- run summary table

Output files:

- `artifacts/data_experiments/charts/error_fallback_summary.png`
- `artifacts/data_experiments/tables/error_fallback_summary.csv`
- `artifacts/data_experiments/outputs/error_cases.json`

## Recommended first chart pack

For the first reproducible experiment, implement these first:

1. `temperature_prediction_comparison.png`
2. `displacement_magnitude_comparison.png`
3. `material_comparison_dashboard.png`
4. `model_disagreement_heatmap.png`
5. `prediction_vs_time_travel_time.png`
6. `error_fallback_summary.png`

This set is enough to:

- compare the four models
- compare materials
- show sensitivity to inputs
- show disagreement
- show operational readiness and fallback behavior

## Suggested output formats

Charts:

- `.png` for quick README and presentation use
- optional `.html` for interactive dashboards later

Tables and metrics:

- `.csv` for flat summary metrics
- `.json` for detailed structured records

Recommended structure:

```text
artifacts/data_experiments/
  inputs/
  outputs/
  tables/
  charts/
  reports/
```
