# Model Comparison Analytics

This folder contains a lightweight analytics pipeline for comparing MeshGraphNet, FNO, and PINN predictions across the four geological media:

- `granite`
- `limestone`
- `sandstone_medium`
- `basalt`

The scripts do not change backend/frontend contracts. They call the existing backend prediction endpoint and turn responses into comparison tables, charts, and an HTML dashboard.

## 1. Generate Scenarios

```bash
python3 analytics/scripts/generate_prediction_scenarios.py
```

Output:

```text
analytics/prediction_scenarios/scenarios_all_rocks.json
```

The generated grid includes baseline, temperature, pressure, frequency, and temperature-pressure sweeps for every rock and model.

## 2. Run Predictions Through Backend

Start the stack first:

```bash
docker compose up --build
```

Then run:

```bash
python3 analytics/scripts/run_model_comparison_predictions.py \
  --backend-url http://127.0.0.1:8000
```

Outputs:

```text
analytics/outputs/model_comparison_predictions.json
analytics/outputs/model_comparison_metrics.csv
```

For a quick smoke run:

```bash
python3 analytics/scripts/run_model_comparison_predictions.py \
  --backend-url http://127.0.0.1:8000 \
  --limit 12
```

## 3. Generate Charts

```bash
python3 analytics/scripts/generate_model_comparison_charts.py
```

Outputs:

```text
analytics/charts/azimuth_comparison.svg
analytics/charts/travel_time_comparison.svg
analytics/charts/magnitude_comparison.svg
analytics/charts/temperature_sensitivity.svg
analytics/charts/pressure_sensitivity.svg
analytics/charts/frequency_sensitivity.svg
analytics/charts/direction_components.svg
analytics/charts/temperature_pressure_heatmap.svg
analytics/charts/travel_time_3d_surface.html
analytics/charts/model_comparison_dashboard.html
```

The chart generator uses only the Python standard library and writes SVG/HTML artifacts, so it works without installing plotting libraries.

## PINN Data Readiness Reports

Before long training runs, generate data and loss diagnostics:

```bash
PYTHONPATH=pinn-service/src python3 pinn-service/scripts/generate_data_quality_report.py

PYTHONPATH=pinn-service/src python3 pinn-service/scripts/create_train_val_split.py \
  --val-fraction 0.1 \
  --seed 42

PYTHONPATH=pinn-service/src python3 pinn-service/scripts/estimate_loss_scales.py \
  --dataset pinn-service/artifacts/rod_experiments/splits/train_samples.npz \
  --sample-limit 8192 \
  --batch-size 512 \
  --device cpu
```

Expected outputs:

```text
pinn-service/artifacts/rod_experiments/reports/data_quality_report.json
pinn-service/artifacts/rod_experiments/reports/data_quality_report.html
pinn-service/artifacts/rod_experiments/splits/train_samples.npz
pinn-service/artifacts/rod_experiments/splits/val_samples.npz
pinn-service/artifacts/rod_experiments/splits/split_metadata.json
pinn-service/artifacts/rod_experiments/reports/loss_scale_report.json
pinn-service/artifacts/rod_experiments/reports/loss_scale_report.html
```
