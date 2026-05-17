# Model Comparison Final Report

## Inputs inspected
- Summary file: `/Users/askarovi/Documents/New project/artifacts/data_experiments/results_2d_4materials_balanced/summary_2d.csv`
- Material source: `/Users/askarovi/Documents/New project/backend/data/media/catalog.json`

## Dataset usage
- Rows loaded: `160`
- Rows used after 2D validation: `160`
- Materials included: `basalt, sandstone, granite, limestone`
- Models included: `pinn, mgn, fno, transformer`

## Availability
- Valid 2D dataset: `yes`
- Reference / ground-truth available: `no`
- Speed / latency available: `no`

## Generated graphs
- PNG graphs created: `11`
- `agreement_by_material_and_model.png`
- `pairwise_model_difference_heatmap.png`
- `error_or_deviation_vs_density.png`
- `error_or_deviation_vs_young_modulus.png`
- `error_or_deviation_vs_thermal_conductivity.png`
- `directional_error_or_deviation_by_azimuth.png`
- `outlier_count_by_model.png`
- `feature_sensitivity_heatmap.png`
- `max_displacement_without_fno.png`
- `temperature_perturbation_without_fno.png`
- `fno_scale_outlier_diagnostic.png`

## Skipped graph items
- Count: `5`
- Graph `inference_time_by_model` skipped because no latency/inference-time column was found in the current summary dataset.
- Graph `speed_vs_accuracy_tradeoff` skipped because no latency/inference-time column and no reference target columns were found.
- Graph `speed_vs_consistency_tradeoff` skipped because no latency/inference-time column was found in the current summary dataset.
- Tables `model_accuracy_summary.csv`, `error_by_material_model.csv`, and `error_by_output_field.csv` were skipped because no explicit ground-truth or target columns were available.
- Graphs `error_by_model_and_output_field` and `error_by_material_and_model` were skipped because no explicit ground-truth or target columns were available.

## FNO warnings
- FNO produces scale outliers relative to the other models; treat these results as scale-unstable prototype outputs rather than validated physical displacements or temperatures.

## Chapter 5 usage note
Use these figures as comparative prototype results for identical 2D source-probe scenarios.
They support discussion of model agreement, stability, directional behavior, and observed feature-output associations.
They should not be described as fully validated field-scale thermoelastic simulations.
