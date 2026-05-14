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
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
```

If `python` is not found, use:

```powershell
py scripts/generate_experiment_inputs.py `
  --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
```

Default grid:

- materials: `sandstone_medium`, `basalt`
- temperatures: `20, 80, 140, 220, 300`
- pressures: `5, 35`
- time points: `6, 12`

That yields `40` cases by default.

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
  --output-dir artifacts/data_experiments/charts
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

### Charts

Charts are saved here:

```text
artifacts/data_experiments/charts/
```

Expected chart files:

- `temperature_comparison.png`
- `displacement_components_comparison.png`
- `displacement_magnitude_comparison.png`
- `material_comparison_sandstone_vs_basalt.png`
- `basalt_vs_sandstone_travel_time.png`
- `basalt_vs_sandstone_displacement.png`
- `model_disagreement.png`
- `prediction_vs_time.png`
- `service_status_summary.png`

## One-line versions

If you prefer one-line PowerShell commands:

```powershell
python scripts/generate_experiment_inputs.py --output artifacts/data_experiments/inputs/model_comparison_inputs.jsonl
```

```powershell
python scripts/run_model_service_experiment.py --input artifacts/data_experiments/inputs/model_comparison_inputs.jsonl --output-dir artifacts/data_experiments/results --timeout-seconds 120
```

```powershell
python scripts/generate_model_comparison_charts.py --input artifacts/data_experiments/results/summary.csv --output-dir artifacts/data_experiments/charts
```

## Smoke run

If you want to test the whole pipeline on one case first:

```powershell
python scripts/generate_experiment_inputs.py `
  --output artifacts/data_experiments/inputs/smoke_case.jsonl `
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
  --output-dir artifacts/data_experiments/charts-smoke
```

Smoke outputs go here:

```text
artifacts/data_experiments/results-smoke/
artifacts/data_experiments/charts-smoke/
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
