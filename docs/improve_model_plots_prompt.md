# Prompt for Codex: Improve model comparison plots, fix FNO fallback/3D issues, and add heatmaps

You are a senior ML engineer, scientific visualization engineer, and backend reviewer.

Repository:

```text
https://github.com/askarov32/ai-directional-prediction
```

The project compares several model services for AI directional prediction of thermoelastic wave propagation in geological media:

- PINN
- MGN
- FNO
- Transformer

Current output data is stored in `summary.csv`.
Current plots compare model responses for basalt and sandstone cases.

However, the current visualization and FNO behavior have several problems.

---

## Main problems to fix

### 1. FNO is currently running in fallback mode

FNO was switched to fallback.

Do not present FNO fallback results as if they were real trained FNO predictions.

Required behavior:

- Detect `fallback_used == true`.
- Show fallback status explicitly in all plots.
- Rename labels like:
  - `fno` → `fno (fallback)`
  - `mgn` → `mgn (fallback)` if applicable.
- In scientific comparison plots, either:
  - exclude fallback models by default, or
  - include them in a separate section called “Fallback / diagnostic responses”.
- Add CLI/config option:

```bash
--include-fallback true|false
```

Default should be:

```bash
--include-fallback false
```

Fallback responses should never silently appear in the main comparison.

---

### 2. Fix FNO 3D handling

Current FNO appears to use a 3D-to-2D domain adaptation mode, for example:

```text
rect_3d_to_rect_2d
```

This is not acceptable for full 3D directional prediction unless it is explicitly documented as a fallback/adapter mode.

Audit the FNO service and fix its 3D behavior.

Check:

- FNO request schema.
- FNO input tensor construction.
- How `source_x`, `source_y`, `source_z`, `probe_x`, `probe_y`, `probe_z` are used.
- Whether `z` is ignored or collapsed.
- Whether output direction has a real `direction_z`.
- Whether elevation is derived from real 3D output or hardcoded/defaulted.
- Whether `effective_domain_adaptation` is hiding a 3D/2D mismatch.

Required outcome:

- If FNO is intended to be 3D, implement proper 3D input handling.
- If FNO is currently only 2D, make this explicit:
  - mark it as `fno_2d_fallback`;
  - do not compare it against 3D models as a real 3D predictor;
  - document this limitation.
- Add validation that rejects or clearly marks requests where a 3D case is passed to a 2D-only FNO model.

Add tests for:

- FNO receives non-zero `source_z` and `probe_z`.
- FNO output contains meaningful `direction_z`.
- FNO elevation is not always zero unless physically justified.
- FNO fallback is correctly marked in response metadata.

---

### 3. Investigate invalid FNO scale

Some plots show physically unrealistic FNO values, for example:

```text
max_displacement ≈ 1.78e+07
max_temperature_perturbation ≈ 2.47e+06
```

This likely indicates one of these problems:

- missing denormalization;
- double denormalization;
- wrong units;
- fallback output using placeholder values;
- plotting stale data;
- mixing old and new `summary.csv`;
- reading wrong columns;
- using failed/fallback responses in successful-response plots.

Required actions:

- Trace where `max_displacement` and `max_temperature_perturbation` are produced.
- Check model response JSON before aggregation.
- Check the generated `summary.csv`.
- Check plotting script aggregation logic.
- Add sanity bounds and warnings.

Example sanity checks:

```python
if max_displacement > DISPLACEMENT_SANITY_LIMIT:
    mark_as_outlier = True

if max_temperature_perturbation > TEMPERATURE_SANITY_LIMIT:
    mark_as_outlier = True
```

Do not silently plot invalid values on normal linear scale.

For outliers:

- annotate them;
- optionally move them to a separate “Outlier diagnostics” plot;
- use log-scale only when the chart title clearly says so.

---

## Plot improvements

Refactor the plotting code into clean, reusable functions.

Suggested structure:

```text
scripts/
  analyze_model_outputs.py
  plot_model_comparison.py
  plot_heatmaps.py
  generate_report.py

reports/
  figures/
  model_comparison_report.md
```

Every plot should have:

- clear title;
- units;
- model status in legend;
- fallback/checkpoint distinction;
- consistent model ordering:
  1. `pinn`
  2. `mgn`
  3. `fno`
  4. `transformer`
- consistent material ordering:
  1. `basalt`
  2. `sandstone`
- readable labels;
- no misleading scientific comparison between real models and fallback models.

---

## Required improved plots

### A. Service status summary

Improve the current service status plot.

It should show:

- `ok_checkpoint`
- `ok_fallback`
- `error`
- `timeout`

Use stacked bars per model.

The chart should answer:

> Which model produced real checkpoint predictions and which model returned fallback responses?

---

### B. Model validity summary

Add a new plot:

```text
Model validity summary
```

For each model, show:

- total requests;
- successful checkpoint responses;
- fallback responses;
- failed responses;
- outlier responses.

This should make it impossible to miss that FNO is fallback.

---

### C. Travel time comparison by material

Improve current chart:

```text
Travel-time comparison by material
```

Rules:

- Exclude fallback by default.
- If fallback is included, mark it visually and in labels.
- Show mean ± standard deviation or mean ± confidence interval.
- Use grouped bars or boxplots.
- Add sample count `n`.
- Avoid comparing models with different service modes without annotation.

---

### D. Max displacement comparison

Current displacement plot is broken by extreme FNO values.

Improve it:

- Create one normal-scale plot for valid non-outlier responses.
- Create one log-scale diagnostic plot for all responses.
- Add labels for outliers.
- Add warning in report if FNO fallback dominates scale.

Required plots:

```text
max_displacement_valid_only.png
max_displacement_log_diagnostic.png
```

---

### E. Temperature perturbation comparison

Same rules as displacement.

Required plots:

```text
temperature_perturbation_valid_only.png
temperature_perturbation_log_diagnostic.png
```

---

### F. Direction vector comparison

Current direction component plot is too aggregated and can be misleading.

Improve it:

- Normalize direction vectors before comparison.
- Validate that direction vectors are unit vectors or close to unit vectors.
- Plot mean direction components by model.
- Add separate angular comparison:
  - azimuth;
  - elevation;
  - angular difference between models.

Do not use raw component averages alone as the main directional quality metric.

---

### G. Fix azimuth disagreement

Current azimuth disagreement plot likely uses normal standard deviation, which is wrong for angles.

Replace ordinary angular standard deviation with circular statistics.

Implement circular mean/std for azimuth.

Use a proper circular distance:

```python
delta = abs((a - b + 180) % 360 - 180)
```

Required plot:

```text
azimuth_circular_disagreement_by_case.png
```

The plot should show cases where models strongly disagree.

---

### H. Elevation comparison

Current FNO elevation appears to be zero.

Improve plot:

- show elevation by model;
- show fallback status;
- flag models where elevation is always zero;
- add warning if this comes from 2D adaptation or missing `direction_z`.

Required report warning:

```text
FNO elevation is always zero. This suggests 2D fallback/adaptation or missing 3D direction output.
```

---

### I. Depth sensitivity

Current depth sensitivity uses too few points.

Improve it:

- generate more probe depth values, for example:

```text
probe_z = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
```

- run the same controlled cases for basalt and sandstone;
- keep source, temperature, pressure, time, and material constant;
- plot travel time and displacement vs depth.

Required plots:

```text
depth_sensitivity_travel_time.png
depth_sensitivity_displacement.png
depth_sensitivity_temperature.png
```

If only two depth points are available, mark the plot as:

```text
limited diagnostic: only 2 depth samples
```

---

## Add heatmaps

Add heatmaps because they are more useful for model comparison across many cases.

### 1. Case × model heatmap: travel time

Rows:

```text
case_id
```

Columns:

```text
model
```

Values:

```text
travel_time_ms_pred
```

Required output:

```text
heatmap_case_model_travel_time.png
```

Use fallback annotations in cell labels or column labels.

---

### 2. Case × model heatmap: max displacement

Rows:

```text
case_id
```

Columns:

```text
model
```

Values:

```text
max_displacement
```

Required output:

```text
heatmap_case_model_displacement.png
```

If values span many orders of magnitude, use log scale:

```python
log10(value)
```

Title must say:

```text
log10(max displacement)
```

---

### 3. Case × model heatmap: temperature perturbation

Rows:

```text
case_id
```

Columns:

```text
model
```

Values:

```text
max_temperature_perturbation
```

Required output:

```text
heatmap_case_model_temperature.png
```

Use log scale if needed.

---

### 4. Model disagreement heatmap

Create pairwise model disagreement heatmaps.

For each metric:

- travel time;
- max displacement;
- temperature perturbation;
- azimuth;
- elevation.

Rows and columns:

```text
model × model
```

Values:

```text
mean absolute difference
```

For azimuth use circular angular distance, not normal subtraction.

Required outputs:

```text
heatmap_model_disagreement_travel_time.png
heatmap_model_disagreement_displacement.png
heatmap_model_disagreement_temperature.png
heatmap_model_disagreement_azimuth.png
heatmap_model_disagreement_elevation.png
```

---

### 5. Material × model heatmap

Rows:

```text
material
```

Columns:

```text
model
```

Values:

- mean travel time;
- mean max displacement;
- mean temperature perturbation.

Required outputs:

```text
heatmap_material_model_travel_time.png
heatmap_material_model_displacement.png
heatmap_material_model_temperature.png
```

---

### 6. Parameter sensitivity heatmap

Create heatmaps showing how predictions vary with input parameters.

Possible rows/columns:

- `time_ms × model`
- `probe_z × model`
- `temperature_c × model`
- `pressure_mpa × model`
- `material × model`

Metrics:

- travel time;
- max displacement;
- temperature perturbation.

Required outputs:

```text
heatmap_time_model_travel_time.png
heatmap_probe_z_model_travel_time.png
heatmap_temperature_model_temperature_perturbation.png
heatmap_pressure_model_displacement.png
```

Only generate a heatmap if there are enough unique values.
If there are not enough unique values, skip it and write a warning in the report.

---

## Report generation

Generate a Markdown report:

```text
reports/model_comparison_report.md
```

The report must include:

1. Dataset summary.
2. Number of cases.
3. Number of responses per model.
4. Number of fallback responses per model.
5. Number of checkpoint responses per model.
6. Number of errors/timeouts per model.
7. List of outlier cases.
8. Key observations.
9. Limitations.
10. All generated plots.

The report should clearly state:

- FNO is currently fallback, not a real trained 3D FNO prediction.
- FNO 3D handling must be fixed before scientific comparison.
- Any fallback model should be treated as diagnostic only.
- Any outlier values should not be interpreted as physically valid until scaling/normalization is verified.

---

## Data validation

Add validation before plotting.

Check required columns:

```text
case_id
model
status
service_mode
fallback_used
material
temperature_c
pressure_mpa
time_ms
source_x
source_y
source_z
probe_x
probe_y
probe_z
direction_x
direction_y
direction_z
azimuth_deg
elevation_deg
magnitude
travel_time_ms_pred
max_displacement
max_temperature_perturbation
wave_type
model_version
http_status
```

If required columns are missing, fail with a clear error.

Add derived columns:

```python
is_checkpoint = service_mode == "checkpoint" and fallback_used == False
is_fallback = fallback_used == True or service_mode == "fallback"
is_error = status != "ok" or http_status >= 400
model_label = model + " (fallback)" if is_fallback else model
direction_norm = sqrt(direction_x**2 + direction_y**2 + direction_z**2)
```

Add warnings if:

- `direction_norm` is far from 1;
- `elevation_deg` is always 0 for a model;
- `direction_z` is always 0 for a model;
- `max_displacement` has extreme outliers;
- `max_temperature_perturbation` has extreme outliers;
- a model has only fallback responses.

---

## CLI interface

Add or improve CLI:

```bash
python scripts/generate_model_report.py \
  --input summary.csv \
  --output-dir reports \
  --include-fallback false \
  --save-svg true \
  --save-png true
```

Expected output:

```text
reports/
  model_comparison_report.md
  figures/
    service_status_summary.png
    model_validity_summary.png
    travel_time_by_material.png
    max_displacement_valid_only.png
    max_displacement_log_diagnostic.png
    temperature_perturbation_valid_only.png
    temperature_perturbation_log_diagnostic.png
    direction_components_by_model.png
    azimuth_circular_disagreement_by_case.png
    elevation_by_model.png
    depth_sensitivity_travel_time.png
    depth_sensitivity_displacement.png
    heatmap_case_model_travel_time.png
    heatmap_case_model_displacement.png
    heatmap_case_model_temperature.png
    heatmap_model_disagreement_travel_time.png
    heatmap_model_disagreement_displacement.png
    heatmap_model_disagreement_temperature.png
    heatmap_model_disagreement_azimuth.png
    heatmap_material_model_travel_time.png
```

---

## Tests

Add tests for:

1. Fallback detection.
2. Excluding fallback models by default.
3. Including fallback models only when `--include-fallback true`.
4. Circular azimuth difference.
5. Direction vector norm validation.
6. Outlier detection.
7. Heatmap generation.
8. Report generation.
9. FNO 3D request validation.
10. FNO 2D fallback marking.

---

## Acceptance criteria

The task is complete only if:

- FNO fallback is clearly marked everywhere.
- Fallback models are excluded from scientific plots by default.
- FNO 3D behavior is audited and either fixed or explicitly marked as unsupported.
- No plot silently mixes checkpoint predictions and fallback predictions.
- Extreme FNO values do not break normal comparison plots.
- Heatmaps are added.
- Circular statistics are used for azimuth disagreement.
- A Markdown report is generated.
- All new plotting and validation logic has tests.
- Documentation explains how to reproduce the plots from `summary.csv`.

---

## Important interpretation note

Do not simply “make plots prettier”.

The main goal is to make the analysis scientifically honest:

- separate checkpoint predictions from fallback predictions;
- separate valid responses from outliers;
- avoid misleading comparison of fallback FNO with real model outputs;
- fix or explicitly document the current FNO 3D limitation;
- make all plots reproducible from `summary.csv`.
