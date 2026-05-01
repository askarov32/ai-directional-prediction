# Granite Analytics

This directory is an isolated analytics bundle for prediction sweeps. It does not modify the main frontend, layout, styles, or application pages.

Even though the bundle is named `granite-analytics`, the scripts are medium-parameterized:

- default medium: `granite`
- default model: `pinn`
- override example: `--medium-id basalt`

## What is inside

- `inputs/granite_scenarios.json`
  Scenario pack with baseline request, representative cases, and sweep definitions.
- `scripts/run_granite_predictions.py`
  Calls the live backend and saves all prediction responses into one JSON artifact.
- `scripts/generate_granite_charts.py`
  Builds charts, summary metrics, a 3D artifact, and the HTML report from the saved predictions.
- `outputs/`
  Machine-readable prediction and summary artifacts.
- `charts/`
  PNG charts and a standalone 3D HTML surface.
- `granite_analytics_report.html`
  Human-readable analytics report.

## Expected workflow

1. Start the stack:

```bash
docker compose up --build
```

2. Run the prediction sweep:

```bash
python3 granite-analytics/scripts/run_granite_predictions.py --medium-id granite
```

3. Generate charts and the report:

```bash
python3 granite-analytics/scripts/generate_granite_charts.py
```

## Medium override

The analytics package is not hardcoded to granite in the scripts. The input medium can be changed at runtime:

```bash
python3 granite-analytics/scripts/run_granite_predictions.py --medium-id quartzite
```

The output files still stay inside `granite-analytics/`, which makes it easy to compare multiple runs while keeping the main app untouched.

## Produced artifacts

- `outputs/granite_predictions.json`
  Full set of resolved requests and normalized backend responses.
- `outputs/granite_metrics_summary.json`
  Aggregated metrics and quick findings.
- `charts/amplitude_time_series.png`
  Displacement vs time for multiple amplitudes.
- `charts/metrics_comparison.png`
  Comparison of representative scenarios.
- `charts/temperature_sensitivity.png`
  Temperature sweep.
- `charts/pressure_sensitivity.png`
  Pressure sweep.
- `charts/frequency_sensitivity.png`
  Frequency sweep.
- `charts/temperature_pressure_heatmap.png`
  Displacement heatmap over the temperature-pressure grid.
- `charts/granite_3d_surface.png`
  3D surface of elevation response.
- `charts/granite_3d_surface.html`
  Standalone HTML version of the 3D surface.
- `granite_analytics_report.html`
  Final visual report.

## Notes

- The scripts use the existing backend contract at `/api/v1/predictions`.
- They assume the backend is reachable at `http://127.0.0.1:8000/api/v1` unless overridden.
- The current charts are designed for the `PINN` route, but the request format remains unified.
