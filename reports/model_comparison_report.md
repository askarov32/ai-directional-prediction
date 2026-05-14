# Model Comparison Report

## Dataset summary
- Input file: `artifacts/data_experiments/results/summary.csv`
- Case count: `40`
- Response count: `160`
- Include fallback in scientific plots: `False`

## Response summary by model
| Model | Total | Checkpoint | Fallback | Error | Timeout | Outlier |
|---|---:|---:|---:|---:|---:|---:|
| `pinn` | 40 | 0 | 0 | 40 | 0 | 0 |
| `mgn` | 40 | 0 | 0 | 40 | 0 | 0 |
| `fno` | 40 | 40 | 0 | 0 | 0 | 40 |
| `transformer` | 40 | 40 | 0 | 0 | 0 | 0 |

## Warnings
- fno elevation is always zero. This suggests 2D fallback/adaptation or missing 3D direction output.
- fno direction_z is always zero.
- fno has displacement outliers above sanity limit 1.
- fno has temperature perturbation outliers above sanity limit 10000.

## Outlier cases
- `case_001_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_002_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_003_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470058.75`
- `case_004_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_005_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_006_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_007_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_008_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_009_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_010_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_011_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_012_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_013_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_014_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_015_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_016_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_017_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_018_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_019_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_020_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_021_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_022_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470058.75`
- `case_023_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_024_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_025_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_026_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470058.75`
- `case_027_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- `case_028_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790334.0`, max_temperature_perturbation = `2470045.0`
- `case_029_basalt` / `fno` / `basalt`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790338.0`, max_temperature_perturbation = `2470057.25`
- `case_030_sandstone` / `fno` / `sandstone`: reasons = `max_displacement, max_temperature_perturbation`, max_displacement = `17790332.0`, max_temperature_perturbation = `2470058.75`
- ... and 10 more outlier rows.

## Key observations
- Fallback responses were excluded from scientific plots by default.
- FNO elevation is always zero in the analyzed run, which strongly suggests 2D adaptation rather than true 3D directional prediction.
- FNO produces displacement outliers and should not be interpreted as physically calibrated until scaling is verified.
- FNO produces temperature perturbation outliers, so raw cross-model scale comparisons remain diagnostic only.
- Circular statistics are used for azimuth disagreement, so wrap-around effects near ±180 degrees are handled correctly.

## Limitations
- Any fallback model must be treated as diagnostic only and not as a scientific comparator.
- Any outlier values should not be interpreted as physically valid until scaling and normalization are verified.
- These plots summarize service outputs, not ground-truth error against laboratory or COMSOL reference targets.
- FNO is currently operating with `rect_3d_to_rect_2d` adaptation on these runs, so it is not a full 3D predictor yet.

## Chart generation details
- Generated figures: `35`
- Skipped parameter heatmaps: `none`

## Generated plots

## azimuth_circular_disagreement_by_case.png
![azimuth_circular_disagreement_by_case.png](figures/azimuth_circular_disagreement_by_case.png)

## basalt_vs_sandstone_displacement.png
![basalt_vs_sandstone_displacement.png](figures/basalt_vs_sandstone_displacement.png)

## basalt_vs_sandstone_travel_time.png
![basalt_vs_sandstone_travel_time.png](figures/basalt_vs_sandstone_travel_time.png)

## depth_sensitivity.png
![depth_sensitivity.png](figures/depth_sensitivity.png)

## depth_sensitivity_displacement.png
![depth_sensitivity_displacement.png](figures/depth_sensitivity_displacement.png)

## depth_sensitivity_temperature.png
![depth_sensitivity_temperature.png](figures/depth_sensitivity_temperature.png)

## depth_sensitivity_travel_time.png
![depth_sensitivity_travel_time.png](figures/depth_sensitivity_travel_time.png)

## displacement_components_comparison.png
![displacement_components_comparison.png](figures/displacement_components_comparison.png)

## displacement_magnitude_comparison.png
![displacement_magnitude_comparison.png](figures/displacement_magnitude_comparison.png)

## domain_adaptation_summary.png
![domain_adaptation_summary.png](figures/domain_adaptation_summary.png)

## elevation_comparison.png
![elevation_comparison.png](figures/elevation_comparison.png)

## heatmap_case_model_displacement.png
![heatmap_case_model_displacement.png](figures/heatmap_case_model_displacement.png)

## heatmap_case_model_temperature.png
![heatmap_case_model_temperature.png](figures/heatmap_case_model_temperature.png)

## heatmap_case_model_travel_time.png
![heatmap_case_model_travel_time.png](figures/heatmap_case_model_travel_time.png)

## heatmap_material_model_displacement.png
![heatmap_material_model_displacement.png](figures/heatmap_material_model_displacement.png)

## heatmap_material_model_temperature.png
![heatmap_material_model_temperature.png](figures/heatmap_material_model_temperature.png)

## heatmap_material_model_travel_time.png
![heatmap_material_model_travel_time.png](figures/heatmap_material_model_travel_time.png)

## heatmap_model_disagreement_azimuth.png
![heatmap_model_disagreement_azimuth.png](figures/heatmap_model_disagreement_azimuth.png)

## heatmap_model_disagreement_displacement.png
![heatmap_model_disagreement_displacement.png](figures/heatmap_model_disagreement_displacement.png)

## heatmap_model_disagreement_elevation.png
![heatmap_model_disagreement_elevation.png](figures/heatmap_model_disagreement_elevation.png)

## heatmap_model_disagreement_temperature.png
![heatmap_model_disagreement_temperature.png](figures/heatmap_model_disagreement_temperature.png)

## heatmap_model_disagreement_travel_time.png
![heatmap_model_disagreement_travel_time.png](figures/heatmap_model_disagreement_travel_time.png)

## heatmap_pressure_model_displacement.png
![heatmap_pressure_model_displacement.png](figures/heatmap_pressure_model_displacement.png)

## heatmap_probe_z_model_travel_time.png
![heatmap_probe_z_model_travel_time.png](figures/heatmap_probe_z_model_travel_time.png)

## heatmap_temperature_model_temperature_perturbation.png
![heatmap_temperature_model_temperature_perturbation.png](figures/heatmap_temperature_model_temperature_perturbation.png)

## heatmap_time_model_travel_time.png
![heatmap_time_model_travel_time.png](figures/heatmap_time_model_travel_time.png)

## material_comparison_sandstone_vs_basalt.png
![material_comparison_sandstone_vs_basalt.png](figures/material_comparison_sandstone_vs_basalt.png)

## max_displacement_log_diagnostic.png
![max_displacement_log_diagnostic.png](figures/max_displacement_log_diagnostic.png)

## max_displacement_valid_only.png
![max_displacement_valid_only.png](figures/max_displacement_valid_only.png)

## model_validity_summary.png
![model_validity_summary.png](figures/model_validity_summary.png)

## prediction_vs_time.png
![prediction_vs_time.png](figures/prediction_vs_time.png)

## service_status_summary.png
![service_status_summary.png](figures/service_status_summary.png)

## temperature_comparison.png
![temperature_comparison.png](figures/temperature_comparison.png)

## temperature_perturbation_log_diagnostic.png
![temperature_perturbation_log_diagnostic.png](figures/temperature_perturbation_log_diagnostic.png)

## temperature_perturbation_valid_only.png
![temperature_perturbation_valid_only.png](figures/temperature_perturbation_valid_only.png)
