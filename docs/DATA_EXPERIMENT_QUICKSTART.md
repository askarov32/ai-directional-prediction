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

Default example with 20 cases:

```bash
python3 scripts/generate_experiment_inputs.py \
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --num-cases 20
```

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
  --output-dir artifacts/data_experiments/results
```

If your machine uses different published ports, pass them explicitly.

Example for a machine where `pinn-service` is published on `9013` and `mgn-service` on `9011`:

```bash
python3 scripts/run_model_service_experiment.py \
  --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl \
  --output-dir artifacts/data_experiments/results \
  --pinn-url http://localhost:9013 \
  --mgn-url http://localhost:9011 \
  --fno-url http://localhost:9002 \
  --transformer-url http://localhost:9004
```

## Step 3. Generate charts

```bash
python3 scripts/generate_model_comparison_charts.py \
  --input artifacts/data_experiments/results/summary.csv \
  --output-dir artifacts/data_experiments/charts
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

Current chart files:

- `temperature_comparison.png`
- `displacement_components_comparison.png`
- `displacement_magnitude_comparison.png`
- `material_comparison_sandstone_vs_basalt.png`
- `model_disagreement.png`
- `prediction_vs_time.png`
- `service_status_summary.png`

## What the result files mean

### `summary.csv`

This is the main flat table for analysis and chart building.

It includes:

- `case_id`
- `model`
- `status`
- `service_mode`
- `fallback_used`
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

To make this chart meaningful, generate a multi-time input grid in the next experiment run.

## Recommended first smoke run

Generate one case:

```bash
python3 scripts/generate_experiment_inputs.py \
  --output artifacts/data_experiments/inputs/smoke_case.jsonl \
  --num-cases 1
```

Run the experiment:

```bash
python3 scripts/run_model_service_experiment.py \
  --input artifacts/data_experiments/inputs/smoke_case.jsonl \
  --output-dir artifacts/data_experiments/results-smoke
```

Generate smoke charts:

```bash
python3 scripts/generate_model_comparison_charts.py \
  --input artifacts/data_experiments/results-smoke/summary.csv \
  --output-dir artifacts/data_experiments/charts-smoke
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
