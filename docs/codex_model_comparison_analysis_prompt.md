# Codex Task: Step-by-Step Neural Model Comparison Analysis for Thesis Results

## Goal

Build a complete, reproducible analysis pipeline for Chapter 5 of the bachelor thesis:

```text
AI Directional Prediction of Thermoelastic Wave Propagation in Geological Media
```

The goal is to compare four model services:

```text
PINN
FNO
MeshGraphNet
Transformer-style baseline
```

not only by raw predicted values, but by:

```text
prediction speed
prediction accuracy if reference targets exist
model agreement if no reference targets exist
physical consistency
feature sensitivity
directional behavior
stability and outliers
```

The output should be a set of clean graphs, summary tables, validation reports, and thesis-ready figure captions.

---

# Important Scientific Context

The thesis has the following positioning:

```text
Chapter 1 — physical and geological foundations
Chapter 2 — mathematical formulation of thermoelastic wave propagation
Chapter 3 — methodology of directional prediction
Chapter 4 — implementation and system architecture
Chapter 5 — results and discussion
```

The theoretical formulation is written in general 3D form, but the practical experiments are performed on simplified 2D domains / 2D cross-sections / selected source-probe directions.

Use this interpretation consistently:

```text
The practical results are comparative directional predictions produced by a research prototype.
They are not fully validated field-scale 3D geophysical simulations.
```

Important distinction between models:

```text
PINN is the only explicitly physics-informed component because it uses thermoelastic residuals derived from the governing equations.

FNO, MeshGraphNet, and Transformer-style services use physically meaningful inputs and outputs, but they do not enforce the full thermoelastic PDE system in the same way as PINN.
```

Do not overclaim.

Do not write that all models solve the coupled thermoelastic equations.

---

# Repository

Work in this repository:

```text
https://github.com/askarov32/ai-directional-prediction
```

Inspect the actual repository structure before editing.

Likely relevant folders/files:

```text
README.md
docker-compose.yml
backend/
frontend/
pinn-service/
fno-service/
mgn-service/
mock-meshgraphnet/
transformer-service/
mock-transformer/
scripts/
data/
artifacts/
figures/
tables/
chapters/
backend/data/media/catalog.json
```

If names differ, adapt to the actual repository.

---

# Input Data

Use available experiment files, preferably:

```text
model_comparison_inputs_2d.jsonl
summary_2d.csv
combined_geological_media_parameters.csv
```

If a balanced 4-material result folder exists, inspect it:

```text
figures/results_2d_4materials_balanced/
tables/results_2d_4materials_balanced/
```

Also search for equivalent files:

```text
summary_2d_4materials.csv
summary_2d_balanced.csv
graph_dataset_2d.csv
model_material_summary.csv
model_status_summary.csv
```

Do not invent missing data.

---

# Output Directories

Create or reuse:

```text
figures/model_comparison/
tables/model_comparison/
scripts/
```

All generated graphs should be saved as:

```text
PNG, dpi=300
SVG if easy
```

All tables should be saved as:

```text
CSV
LaTeX where useful
```

All reports should be saved as:

```text
Markdown or TXT
```

---

# Step 1 — Audit Existing Data and Repository

First inspect the repository and report:

```text
1. Which input files exist.
2. Which summary/result files exist.
3. Which material parameter files exist.
4. Which columns are available in the prediction summary.
5. Which columns are available in the material table.
6. Which models are present.
7. Which materials are present.
8. Whether 2D consistency can be verified.
9. Whether inference time / latency is available.
10. Whether ground truth / target / reference data is available.
```

Create:

```text
tables/model_comparison/data_audit_report.md
```

---

# Step 2 — Validate 2D Consistency

Check whether the prediction rows correspond to the thesis 2D setup.

If columns exist, validate:

```text
requested_domain_type = rect_2d
effective_domain_type = rect_2d
domain.type = rect_2d
source_z = 0.0
probe_z = 0.0
direction_z = 0.0
elevation_deg = 0.0
domain_lz = 0.0
domain_nz = 1
```

If some columns are missing, do not fail. Write a warning.

Create:

```text
tables/model_comparison/data_validation_report.md
```

Include:

```text
total rows
valid 2D rows
excluded rows
models
materials
status counts
fallback counts
non-finite values
missing columns
warnings
```

Use only valid rows where possible.

---

# Step 3 — Build Clean Joined Dataset

Join:

```text
prediction summary
+
geological material parameters
```

Join key:

```text
material
```

Normalize material names:

```text
basalt
sandstone
granite
limestone
```

Optionally include other materials if they are present and valid.

Create helper columns if possible:

```text
E_GPa = E_Pa / 1e9
K_GPa = K_Pa / 1e9
mu_GPa = mu_Pa / 1e9
alpha_1e6_K = alpha_1_K * 1e6
```

If porosity is textual, parse it safely only if unambiguous.

Create:

```text
tables/model_comparison/model_comparison_dataset.csv
```

This dataset should contain, where available:

```text
case_id
model
material
status
fallback_used
travel_time_ms_pred
max_displacement
max_temperature_perturbation
magnitude
azimuth_deg or input_azimuth_deg
inference_time_ms
rho_kg_m3
Vp_m_s
Vs_m_s
E_GPa
K_GPa
mu_GPa
k_W_mK
Cp_J_kgK
alpha_1e6_K
porosity_percent
```

Do not invent values. Leave missing values as blank/NaN.

---

# Step 4 — Determine Whether Accuracy Can Be Computed

Search for reference/ground truth columns or files.

Possible reference fields:

```text
temperature_true
temperature_target
target_temperature
T_true
T_target
disp_x_true
disp_y_true
disp_z_true
u_true
v_true
w_true
travel_time_true
reference_travel_time_ms
comsol_temperature
comsol_displacement
```

Possible reference sources:

```text
COMSOL output files
synthetic targets
test dataset
held-out dataset
target fields
```

If reference exists, compute accuracy metrics.

If reference does not exist, do not call the results "accuracy". Use:

```text
model agreement
relative model difference
physical consistency
stability
feature sensitivity
```

Create:

```text
tables/model_comparison/accuracy_availability_report.md
```

The report should clearly state:

```text
Accuracy metrics available: yes/no
Reference source used: ...
If no: explain that only agreement/consistency analysis is possible.
```

---

# Step 5 — If Reference Exists: Compute Accuracy Metrics

If reference data exists, compute the following where possible:

```text
MAE
RMSE
relative L2 error
absolute travel time error
relative travel time error
temperature RMSE
displacement RMSE
direction error in degrees
```

For vector displacement:

```text
displacement_magnitude = sqrt(u^2 + v^2) for 2D
```

For direction:

```text
direction_error_deg = angular difference between predicted direction and reference/input direction
```

Create:

```text
tables/model_comparison/model_accuracy_summary.csv
tables/model_comparison/model_accuracy_summary.tex
tables/model_comparison/error_by_material_model.csv
tables/model_comparison/error_by_output_field.csv
```

---

# Step 6 — If Reference Does Not Exist: Compute Model Agreement Metrics

If no ground truth exists, compute model agreement instead.

For each case_id and material, compare outputs between models.

Use metrics:

```text
pairwise absolute difference
pairwise relative difference
difference from ensemble median
difference from PINN baseline
```

Important:

- PINN can be used as a physics-informed reference baseline, but do not call it ground truth.
- Write "difference from PINN baseline", not "error against PINN truth".

Create:

```text
tables/model_comparison/model_agreement_summary.csv
tables/model_comparison/pairwise_model_difference.csv
tables/model_comparison/difference_from_pinn_baseline.csv
```

---

# Step 7 — Compute Prediction Speed Metrics

Find inference time / latency columns. Possible names:

```text
inference_time_ms
latency_ms
prediction_time_ms
duration_ms
elapsed_ms
request_duration_ms
runtime_ms
```

If no speed column exists, inspect scripts/logs to see if prediction time can be measured. If not, write a report that speed comparison cannot be computed from current data.

If speed data exists, compute:

```text
mean inference time by model
std inference time by model
median inference time
min/max inference time
p95 inference time if enough samples
```

Create:

```text
tables/model_comparison/model_speed_summary.csv
tables/model_comparison/model_speed_summary.tex
```

---

# Step 8 — Compute Stability and Outlier Metrics

For each model, compute:

```text
number of rows
number of ok rows
number of failed rows
fallback count
non-finite output count
scale outlier count
mean/std of main outputs
coefficient of variation
```

Suggested outlier checks:

```text
abs(max_displacement) > 1e2
abs(max_temperature_perturbation) > 1e4
magnitude > 1e6
NaN / Inf values
```

These are prototype sanity thresholds, not universal physical limits. Document them.

Create:

```text
tables/model_comparison/model_stability_summary.csv
tables/model_comparison/model_stability_summary.tex
```

If FNO is still several orders of magnitude larger than others, explicitly flag it:

```text
FNO scale mismatch / output instability warning
```

---

# Step 9 — Feature Sensitivity Analysis

This is important for the thesis because the models use physical material parameters.

Goal:

```text
Test whether model outputs respond meaningfully to physical input features.
```

If existing predictions cover enough variation, use existing rows.

If not, generate perturbation cases from existing base cases.

Perturb these features when available:

```text
rho_kg_m3
E_GPa
nu
k_W_mK
Cp_J_kgK
alpha_1e6_K
Vp_m_s
Vs_m_s
porosity_percent
```

Suggested perturbation:

```text
-10%
+10%
```

For each model and feature, compute:

```text
relative output change = |y_perturbed - y_base| / (|y_base| + eps)
```

For outputs:

```text
travel_time_ms_pred
max_displacement
max_temperature_perturbation
magnitude
```

If perturbation experiments cannot be run automatically, compute feature-output trends from existing data and clearly state this is not controlled sensitivity.

Create:

```text
tables/model_comparison/feature_sensitivity_summary.csv
```

---

# Step 10 — Generate Required Graphs

Generate graphs only when the required columns exist.

All graph captions must be cautious and thesis-ready.

## Graph 1 — Inference Time by Model

File:

```text
figures/model_comparison/inference_time_by_model.png
figures/model_comparison/inference_time_by_model.svg
```

Type:

```text
box plot or bar plot with mean ± std
```

Use if speed data exists.

Purpose:

```text
Compare practical prediction speed of the four model services.
```

---

## Graph 2 — Speed vs Accuracy Trade-off

File:

```text
figures/model_comparison/speed_vs_accuracy_tradeoff.png
figures/model_comparison/speed_vs_accuracy_tradeoff.svg
```

Type:

```text
scatter plot
```

Axes:

```text
x-axis: mean inference_time_ms
y-axis: mean prediction error
color/label: model
```

Use only if both speed data and reference-based error exist.

If no reference exists, generate:

```text
speed_vs_consistency_tradeoff.png
```

where y-axis is difference from ensemble median or difference from PINN baseline.

---

## Graph 3 — Error by Model and Output Field

File:

```text
figures/model_comparison/error_by_model_and_output_field.png
```

Use only if reference exists.

Axes:

```text
x-axis: model
y-axis: error
hue/group: output field
```

Output fields:

```text
temperature
displacement
travel_time
direction
```

---

## Graph 4 — Error by Material and Model

File:

```text
figures/model_comparison/error_by_material_and_model.png
```

Use only if reference exists.

Axes:

```text
x-axis: material
y-axis: prediction error
hue: model
```

Purpose:

```text
Show whether model accuracy depends on rock type.
```

---

## Graph 5 — Agreement by Material and Model

File:

```text
figures/model_comparison/agreement_by_material_and_model.png
```

Use if no reference exists.

Metric:

```text
difference from ensemble median
or difference from PINN baseline
```

Axes:

```text
x-axis: material
y-axis: agreement deviation
hue: model
```

Purpose:

```text
Compare how far each model deviates from common model behavior or from the physics-informed PINN baseline.
```

---

## Graph 6 — Error or Deviation vs Density

File:

```text
figures/model_comparison/error_or_deviation_vs_density.png
```

Axes:

```text
x-axis: rho_kg_m3
y-axis: error or deviation
hue: model
marker/style: material
```

If no reference exists, use deviation from PINN baseline or ensemble median.

---

## Graph 7 — Error or Deviation vs Young’s Modulus

File:

```text
figures/model_comparison/error_or_deviation_vs_young_modulus.png
```

Axes:

```text
x-axis: E_GPa
y-axis: displacement error or displacement deviation
hue: model
marker/style: material
```

Purpose:

```text
Check whether stiffness influences model accuracy or model disagreement.
```

---

## Graph 8 — Error or Deviation vs Thermal Conductivity

File:

```text
figures/model_comparison/error_or_deviation_vs_thermal_conductivity.png
```

Axes:

```text
x-axis: k_W_mK
y-axis: temperature error or temperature deviation
hue: model
marker/style: material
```

Purpose:

```text
Check whether thermal transport properties influence model behavior.
```

---

## Graph 9 — Directional Error or Directional Deviation vs Input Azimuth

File:

```text
figures/model_comparison/directional_error_or_deviation_by_azimuth.png
```

Axes:

```text
x-axis: input_azimuth_deg
y-axis: error/deviation metric
hue: model
```

Purpose:

```text
Evaluate whether prediction quality or model agreement changes with propagation direction.
```

Important validation:

For each `case_id`, input azimuth should be identical across all models. If not, report the inconsistency and do not use this graph.

---

## Graph 10 — Feature Sensitivity Heatmap

File:

```text
figures/model_comparison/feature_sensitivity_heatmap.png
```

Rows:

```text
rho
E
nu
k
Cp
alpha
Vp
Vs
porosity
```

Columns:

```text
PINN
FNO
MeshGraphNet
Transformer
```

Value:

```text
mean relative output change
```

Purpose:

```text
Show which models react most strongly to physical material features.
```

If controlled perturbations are not available, title should say:

```text
Observed Feature-Output Association
```

not:

```text
Controlled Sensitivity
```

---

## Graph 11 — Outlier Count by Model

File:

```text
figures/model_comparison/outlier_count_by_model.png
```

Axes:

```text
x-axis: model
y-axis: number of warnings/outliers
```

Purpose:

```text
Show model stability and numerical reliability.
```

---

## Graph 12 — Pairwise Model Difference Heatmap

File:

```text
figures/model_comparison/pairwise_model_difference_heatmap.png
```

Use if no reference exists or as additional analysis.

Rows/columns:

```text
PINN
FNO
MeshGraphNet
Transformer
```

Value:

```text
mean absolute difference in selected output
```

Generate separate heatmaps or separate panels only if easy. Otherwise use one main output such as `magnitude` or `max_displacement`.

---

# Step 11 — Handle FNO Carefully

If FNO remains a scale outlier:

1. Do not remove it silently.
2. Create a diagnostic report.
3. Create optional plots excluding FNO for readability.
4. Clearly explain in captions and reports.

Generate:

```text
tables/model_comparison/fno_diagnostic_report.md
```

Optional graphs:

```text
figures/model_comparison/max_displacement_without_fno.png
figures/model_comparison/temperature_perturbation_without_fno.png
figures/model_comparison/fno_scale_outlier_diagnostic.png
```

Correct interpretation:

```text
FNO output values are treated as scale-unstable prototype predictions when they exceed the range of other models by several orders of magnitude.
```

Incorrect interpretation:

```text
FNO physically predicts extremely large real displacement or temperature.
```

---

# Step 12 — Generate Thesis Captions and Interpretation Text

Create:

```text
tables/model_comparison/figure_captions_and_interpretation.md
```

For each graph, include:

```text
figure file
LaTeX includegraphics block
caption
short interpretation paragraph
warnings/limitations
```

Captions must be cautious.

Good wording:

```text
The plot compares model behavior under identical 2D source-probe conditions.
```

```text
The trend suggests a qualitative relationship, but it should not be interpreted as field-validated physical proof.
```

Bad wording:

```text
This graph proves that the model solves thermoelastic wave propagation.
```

---

# Step 13 — Create Main Script

Create one reproducible script:

```text
scripts/generate_model_comparison_analysis.py
```

The script should do:

```text
1. Load prediction summary.
2. Load material parameter table.
3. Normalize names.
4. Validate 2D consistency.
5. Join datasets.
6. Check availability of reference targets.
7. Compute accuracy metrics if possible.
8. Compute model agreement metrics if accuracy is not possible.
9. Compute speed metrics if available.
10. Compute stability/outlier metrics.
11. Compute feature sensitivity/association.
12. Generate graphs.
13. Generate summary tables.
14. Generate captions and reports.
```

The script should run as:

```bash
python scripts/generate_model_comparison_analysis.py
```

Also support optional arguments:

```bash
python scripts/generate_model_comparison_analysis.py   --summary summary_2d.csv   --materials combined_geological_media_parameters.csv   --out-figures figures/model_comparison   --out-tables tables/model_comparison
```

Do not require internet access.

---

# Step 14 — Robustness Requirements

The script must not crash if optional columns are missing.

If a graph cannot be generated, skip it and write the reason into:

```text
tables/model_comparison/skipped_graphs_report.md
```

Examples:

```text
Inference speed graph skipped because no inference_time_ms column was found.
Accuracy graph skipped because no ground truth columns were found.
Feature sensitivity heatmap generated as observed association, not controlled perturbation.
Directional graph skipped because input azimuth is inconsistent across models for same case_id.
```

---

# Step 15 — Final Report

After completing the task, report:

```text
1. Files inspected.
2. Files created.
3. Rows loaded.
4. Rows used.
5. Materials included.
6. Models included.
7. Whether data is valid 2D.
8. Whether reference/ground truth exists.
9. Whether speed data exists.
10. Which graphs were created.
11. Which graphs were skipped and why.
12. Any FNO warnings.
13. Any fallback warnings.
14. Any outlier warnings.
15. How to use the generated graphs in Chapter 5.
```

---

# Expected Generated Files

At minimum:

```text
scripts/generate_model_comparison_analysis.py

tables/model_comparison/data_audit_report.md
tables/model_comparison/data_validation_report.md
tables/model_comparison/model_comparison_dataset.csv
tables/model_comparison/accuracy_availability_report.md
tables/model_comparison/model_stability_summary.csv
tables/model_comparison/model_status_summary.csv
tables/model_comparison/figure_captions_and_interpretation.md
tables/model_comparison/skipped_graphs_report.md
```

If reference exists:

```text
tables/model_comparison/model_accuracy_summary.csv
tables/model_comparison/model_accuracy_summary.tex
tables/model_comparison/error_by_material_model.csv
tables/model_comparison/error_by_output_field.csv
figures/model_comparison/error_by_model_and_output_field.png
figures/model_comparison/error_by_material_and_model.png
```

If no reference exists:

```text
tables/model_comparison/model_agreement_summary.csv
tables/model_comparison/pairwise_model_difference.csv
tables/model_comparison/difference_from_pinn_baseline.csv
figures/model_comparison/agreement_by_material_and_model.png
figures/model_comparison/pairwise_model_difference_heatmap.png
```

If speed data exists:

```text
tables/model_comparison/model_speed_summary.csv
tables/model_comparison/model_speed_summary.tex
figures/model_comparison/inference_time_by_model.png
figures/model_comparison/speed_vs_accuracy_tradeoff.png
```

or, if no reference exists:

```text
figures/model_comparison/speed_vs_consistency_tradeoff.png
```

Always generate where possible:

```text
figures/model_comparison/error_or_deviation_vs_density.png
figures/model_comparison/error_or_deviation_vs_young_modulus.png
figures/model_comparison/error_or_deviation_vs_thermal_conductivity.png
figures/model_comparison/directional_error_or_deviation_by_azimuth.png
figures/model_comparison/outlier_count_by_model.png
```

Optional:

```text
figures/model_comparison/feature_sensitivity_heatmap.png
figures/model_comparison/max_displacement_without_fno.png
figures/model_comparison/temperature_perturbation_without_fno.png
figures/model_comparison/fno_scale_outlier_diagnostic.png
```

---

# Final Reminder

The purpose of this task is not to create decorative plots.

The purpose is to produce a defensible Chapter 5 analysis:

```text
Which model is faster?
Which model is more accurate if reference exists?
If no reference exists, which models agree or disagree?
How do physical material parameters affect model behavior?
Which model is most stable?
Does prediction behavior depend on propagation direction?
Does FNO show scale instability?
```

Use cautious academic interpretation throughout.
