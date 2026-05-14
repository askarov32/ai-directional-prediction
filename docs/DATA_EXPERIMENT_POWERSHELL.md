# Data Experiment PowerShell Guide

This guide is the Windows PowerShell version of the data-experiment pipeline.

Use it if you run the project on Windows and want copy-paste commands without bash syntax.

## Important PowerShell note

In PowerShell:

- multiline continuation uses the backtick: `` ` ``
- not the bash backslash: `\`

So this is wrong in PowerShell:

```powershell
python3 scripts/generate_experiment_inputs.py \
  --output ...
```

And this is correct:

```powershell
python scripts/generate_experiment_inputs.py `
  --output ...
```

## 1. Start the services

If you want only the model services:

```powershell
docker compose up -d pinn-service mgn-service fno-service transformer-service
```

If you want the full stack:

```powershell
docker compose up -d
```

## 2. Generate experiment inputs

Standard 40-case run:

```powershell
python scripts/generate_experiment_inputs.py `
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl `
  --metadata-output artifacts/data_experiments/inputs/model_comparison_inputs.metadata.json
```

If `python` is not found, use:

```powershell
py scripts/generate_experiment_inputs.py `
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
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

Files created:

```text
artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
artifacts/data_experiments/inputs/model_comparison_inputs.metadata.json
```

## 3. Run the direct model-service experiment

If your published Docker ports are the standard ones:

- `pinn-service`: `9003`
- `mgn-service`: `9001`
- `fno-service`: `9002`
- `transformer-service`: `9004`

run:

```powershell
python scripts/run_model_service_experiment.py `
  --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl `
  --output-dir artifacts/data_experiments/results `
  --timeout-seconds 120
```

If your ports are custom, pass them explicitly.

Example:

- `pinn-service` on `9013`
- `mgn-service` on `9011`
- `fno-service` on `9002`
- `transformer-service` on `9004`

```powershell
python scripts/run_model_service_experiment.py `
  --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl `
  --output-dir artifacts/data_experiments/results `
  --timeout-seconds 120 `
  --pinn-url http://localhost:9013 `
  --mgn-url http://localhost:9011 `
  --fno-url http://localhost:9002 `
  --transformer-url http://localhost:9004
```

## 4. Generate charts

```powershell
python scripts/generate_model_comparison_charts.py `
  --input artifacts/data_experiments/results/summary.csv `
  --output-dir artifacts/data_experiments/charts `
  --include-fallback false
```

## 5. Generate markdown report

```powershell
python scripts/generate_model_report.py `
  --input artifacts/data_experiments/results/summary.csv `
  --output-dir reports `
  --include-fallback false `
  --save-png true
```

## Where the files go

### Inputs

```text
artifacts/data_experiments/inputs/
```

Main files:

- `model_comparison_inputs.jsonl`
- `model_comparison_inputs.metadata.json`

### Results

```text
artifacts/data_experiments/results/
```

Main files:

- `raw/requests.jsonl`
- `raw/responses.jsonl`
- `normalized/results.jsonl`
- `summary.csv`
- `summary.json`

Important 3D fields in `summary.csv`:

- `requested_domain_type`
- `effective_domain_type`
- `domain_adaptation`

This lets you see which services ran true `3D` and which were adapted.

### Charts

Charts are saved here:

```text
artifacts/data_experiments/charts/
```

Main chart files now include:

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

The final analysis report is written here:

```text
reports/model_comparison_report.md
```

All report figures are written here:

```text
reports/figures/
```

## One-line versions

If you prefer one-line PowerShell commands:

```powershell
python scripts/generate_experiment_inputs.py --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl --metadata-output artifacts/data_experiments/inputs/model_comparison_inputs.metadata.json
```

```powershell
python scripts/run_model_service_experiment.py --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl --output-dir artifacts/data_experiments/results --timeout-seconds 120
```

```powershell
python scripts/generate_model_comparison_charts.py --input artifacts/data_experiments/results/summary.csv --output-dir artifacts/data_experiments/charts
```

```powershell
python scripts/generate_model_report.py --input artifacts/data_experiments/results/summary.csv --output-dir reports --include-fallback false --save-png true
```

## Smoke run

If you want to test the whole pipeline on one case first:

```powershell
python scripts/generate_experiment_inputs.py `
  --output artifacts/data_experiments/inputs/smoke_case.jsonl `
  --metadata-output artifacts/data_experiments/inputs/smoke_case.metadata.json `
  --num-cases 1
```

```powershell
python scripts/run_model_service_experiment.py `
  --input artifacts/data_experiments/inputs/smoke_case.jsonl `
  --output-dir artifacts/data_experiments/results-smoke `
  --timeout-seconds 120
```

```powershell
python scripts/generate_model_comparison_charts.py `
  --input artifacts/data_experiments/results-smoke/summary.csv `
  --output-dir artifacts/data_experiments/charts-smoke `
  --include-fallback false
```

```powershell
python scripts/generate_model_report.py `
  --input artifacts/data_experiments/results-smoke/summary.csv `
  --output-dir reports-smoke `
  --include-fallback false `
  --save-png true
```

Smoke outputs go here:

```text
artifacts/data_experiments/results-smoke/
artifacts/data_experiments/charts-smoke/
reports-smoke/
```

## Common issues

### PowerShell says `MissingExpressionAfterOperator`

Reason:

- you used bash-style `\` line continuation

Fix:

- use PowerShell backtick `` ` ``
- or write the command in one line

### `python` is not recognized

Try:

```powershell
py scripts/generate_experiment_inputs.py `
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
```

### `pinn-service` returns `503`

Most likely:

- `PINN_DEVICE=cuda`
- but Docker runtime on this machine does not expose CUDA

For a local CPU-only run, change:

```env
PINN_DEVICE=cpu
```

then restart `pinn-service`.

### `fno-service` or `mgn-service` respond, but in fallback mode

This means:

- the service is alive
- but the real checkpoint or dataset path is not ready

The pipeline still records those outputs, but they should be interpreted as fallback/demo responses.

## Related docs

- [Data Experiment Quickstart](/Users/askarovi/Documents/New%20project/docs/DATA_EXPERIMENT_QUICKSTART.md)
- [Model Service Curls](/Users/askarovi/Documents/New%20project/docs/MODEL_SERVICE_CURLS.md)
- [Data Experiment Pipeline](/Users/askarovi/Documents/New%20project/docs/DATA_EXPERIMENT_PIPELINE.md)
