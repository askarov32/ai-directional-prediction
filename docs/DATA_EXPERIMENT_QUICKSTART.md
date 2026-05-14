# Data Experiment Quickstart

This quickstart shows how to run the direct model-service comparison pipeline and where the generated artifacts are written.

## What this pipeline does

The pipeline:

1. generates canonical experiment inputs;
2. sends the same cases directly to:
   - `pinn-service`
   - `mgn-service`
   - `fno-service`
   - `transformer-service`
3. saves raw requests and responses;
4. builds a normalized summary table;
5. generates comparison charts.

## Prerequisites

Start the services first:

```bash
docker compose up -d pinn-service mgn-service fno-service transformer-service
```

If you also want the backend and frontend running:

```bash
docker compose up -d
```

## Step 1. Generate experiment inputs

Default example with 40 cases:

```bash
python3 scripts/generate_experiment_inputs.py \
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --metadata-output artifacts/data_experiments/inputs/model_comparison_inputs.metadata.json
```

Default experiment pool before trimming:

- materials: `sandstone_medium`, `basalt`
- temperatures: `20, 60, 120, 180, 260, 320`
- pressures: `5, 25, 60`
- time points: `4, 8, 12, 16`
- frequencies: `25, 50, 75`
- domain: `rect_3d`
- 4 source variants
- 4 probe variants
- 3 boundary-condition variants

The generator deterministically shuffles that larger pool and keeps `40` cases by default.

This creates:

```text
artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
artifacts/data_experiments/inputs/model_comparison_inputs.metadata.json
```

## Step 2. Run direct model-service experiment

If your Docker ports match the standard setup:

- `pinn-service`: `9003`
- `mgn-service`: `9001`
- `fno-service`: `9002`
- `transformer-service`: `9004`

use:

```bash
python3 scripts/run_model_service_experiment.py \
  --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --output-dir artifacts/data_experiments/results \
  --timeout-seconds 120
```

If your machine uses different published ports, pass them explicitly.

Example for a machine where `pinn-service` is published on `9013` and `mgn-service` on `9011`:

```bash
python3 scripts/run_model_service_experiment.py \
  --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --output-dir artifacts/data_experiments/results \
  --timeout-seconds 120 \
  --pinn-url http://localhost:9013 \
  --mgn-url http://localhost:9011 \
  --fno-url http://localhost:9002 \
  --transformer-url http://localhost:9004
```

## Step 3. Generate charts

```bash
python3 scripts/generate_model_comparison_charts.py \
  --input artifacts/data_experiments/results/summary.csv \
  --output-dir artifacts/data_experiments/charts \
  --include-fallback false
```

## Step 4. Generate markdown report

```bash
python3 scripts/generate_model_report.py \
  --input artifacts/data_experiments/results/summary.csv \
  --output-dir reports \
  --include-fallback false \
  --save-png true
```

## Where files are saved

### Inputs

```text
artifacts/data_experiments/inputs/
```

Main files:

- `model_comparison_inputs.jsonl`
- `model_comparison_inputs.metadata.json`

### Raw and normalized experiment results

```text
artifacts/data_experiments/results/
```

Main files:

- `raw/requests.jsonl`
- `raw/responses.jsonl`
- `normalized/results.jsonl`
- `summary.csv`
- `summary.json`

### Charts

Charts are written here:

```text
artifacts/data_experiments/charts/
```

Current chart files include:

- `temperature_comparison.png`
- `displacement_components_comparison.png`
- `displacement_magnitude_comparison.png`
- `max_displacement_valid_only.png`
- `max_displacement_log_diagnostic.png`
- `temperature_perturbation_valid_only.png`
- `temperature_perturbation_log_diagnostic.png`
- `basalt_vs_sandstone_travel_time.png`
- `basalt_vs_sandstone_displacement.png`
- `elevation_comparison.png`
- `depth_sensitivity.png`
- `depth_sensitivity_travel_time.png`
- `depth_sensitivity_displacement.png`
- `depth_sensitivity_temperature.png`
- `domain_adaptation_summary.png`
- `azimuth_circular_disagreement_by_case.png`
- `prediction_vs_time.png`
- `service_status_summary.png`
- `model_validity_summary.png`
- `heatmap_case_model_travel_time.png`
- `heatmap_case_model_displacement.png`
- `heatmap_case_model_temperature.png`
- `heatmap_model_disagreement_travel_time.png`
- `heatmap_model_disagreement_displacement.png`
- `heatmap_model_disagreement_temperature.png`
- `heatmap_model_disagreement_azimuth.png`
- `heatmap_model_disagreement_elevation.png`
- `heatmap_material_model_travel_time.png`
- `heatmap_material_model_displacement.png`
- `heatmap_material_model_temperature.png`
- `heatmap_time_model_travel_time.png`
- `heatmap_probe_z_model_travel_time.png`
- `heatmap_temperature_model_temperature_perturbation.png`
- `heatmap_pressure_model_displacement.png`

### Markdown report

Final report:

```text
reports/model_comparison_report.md
```

Report figures:

```text
reports/figures/
```

## What the result files mean

### `summary.csv`

This is the main flat table for analysis and chart building.

It includes:

- `case_id`
- `model`
- `status`
- `service_mode`
- `fallback_used`
- `requested_domain_type`
- `effective_domain_type`
- `domain_adaptation`
- `material`
- `temperature_c`
- `pressure_mpa`
- `time_ms`
- `frequency_hz`
- `direction_x`, `direction_y`, `direction_z`
- `azimuth_deg`
- `elevation_deg`
- `magnitude`
- `travel_time_ms_pred`
- `max_displacement`
- `max_temperature_perturbation`
- `wave_type`
- `model_version`
- `error_code`
- `error_message`
- `http_status`

For 3D-first runs this is important:

- `requested_domain_type` is the domain generated by the experiment pack;
- `effective_domain_type` is the domain actually sent to the model service;
- `domain_adaptation` shows whether the runner had to downgrade a model route.

Current example:

- `pinn`, `mgn`, `transformer` can stay on `rect_3d`;
- `fno` is currently downgraded to `rect_2d` with `domain_adaptation=rect_3d_to_rect_2d`.

### `summary.json`

This is the run-level summary:

- number of cases
- number of predict calls
- number of successful results
- number of fallback results
- number of errors
- preflight `/health` and `/ready` snapshots for each service

## Typical issues

### `pinn-service` returns `503`

Most likely reason:

- `PINN_DEVICE=cuda` is set
- but CUDA is not available in the current Docker runtime

For local CPU-only runs, set:

```env
PINN_DEVICE=cpu
```

and restart `pinn-service`.

### `fno-service` or `mgn-service` show fallback mode

This means the service answered, but not from a real ready checkpoint+dataset path.

You will still get outputs, but they are fallback/demo outputs and should be marked as such in analysis.

### `prediction_vs_time.png` looks empty or placeholder-like

This happens if all generated cases use the same `time_ms`.

The current default generator uses a multi-time grid with `4, 8, 12, 16 ms`, so this chart should become populated after a fresh run.

## Recommended first smoke run

Generate one case:

```bash
python3 scripts/generate_experiment_inputs.py \
  --output artifacts/data_experiments/inputs/smoke_case.jsonl \
  --metadata-output artifacts/data_experiments/inputs/smoke_case.metadata.json \
  --num-cases 1
```

Run the experiment:

```bash
python3 scripts/run_model_service_experiment.py \
  --input artifacts/data_experiments/inputs/smoke_case.jsonl \
  --output-dir artifacts/data_experiments/results-smoke \
  --timeout-seconds 120
```

Generate smoke charts:

```bash
python3 scripts/generate_model_comparison_charts.py \
  --input artifacts/data_experiments/results-smoke/summary.csv \
  --output-dir artifacts/data_experiments/charts-smoke \
  --include-fallback false
```

Generate smoke report:

```bash
python3 scripts/generate_model_report.py \
  --input artifacts/data_experiments/results-smoke/summary.csv \
  --output-dir reports-smoke \
  --include-fallback false \
  --save-png true
```

Smoke charts will then be here:

```text
artifacts/data_experiments/charts-smoke/
```

## Related docs

- [Model Service Curls](/Users/askarovi/Documents/New%20project/docs/MODEL_SERVICE_CURLS.md)
- [Data Experiment Charts](/Users/askarovi/Documents/New%20project/docs/DATA_EXPERIMENT_CHARTS.md)
- [Data Experiment Pipeline](/Users/askarovi/Documents/New%20project/docs/DATA_EXPERIMENT_PIPELINE.md)
- [Data Experiment PowerShell Guide](/Users/askarovi/Documents/New%20project/docs/DATA_EXPERIMENT_POWERSHELL.md)
