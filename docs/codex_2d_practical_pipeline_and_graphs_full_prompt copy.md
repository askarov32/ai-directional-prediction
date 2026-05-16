# Codex Task: 2D Practical Pipeline and Result Graphs for Thermoelastic Wave Prediction

This document combines:
1. the practical 2D experiment plan;
2. the graph-generation prompt for Codex.

Use it as one complete prompt for Codex.

---

# Part A — Practical Part Plan

# Practical Part Plan: 2D Model Training, Comparison, and Result Graphs

## 1. Current Status

The thesis has already established the theoretical and mathematical basis:

```text
Chapter 1 — physical and geological foundations
Chapter 2 — mathematical formulation
Chapter 3 — methodology and prediction framework
```

The next stage is the practical part:

```text
Chapter 4 — implementation / experimental setup
Chapter 5 — results, graphs, comparison, and discussion
```

The practical part should not claim a full 3D field-scale solution. The defensible position is:

```text
The mathematical formulation is written in 3D, because real geological media are three-dimensional.
The practical implementation is performed on simplified 2D cross-sections or selected source–probe directions, because full 3D coupled thermoelastic modelling is significantly more complex.
```

So the implementation should be positioned as:

```text
Theory: general 3D formulation
Practice: 2D experimental setup / 2D cross-section / selected propagation paths
```

---

## 2. Uploaded Data Check

The currently available files are:

```text
model_comparison_inputs.jsonl
summary.csv
```

### 2.1 `model_comparison_inputs.jsonl`

This file contains 40 input cases.

Current issue:

```text
requested_domain_type = rect_3d
domain.type = rect_3d
resolution = nx=128, ny=128, nz=48
source_z != 0
probe_z != 0
direction_z != 0
```

Therefore, the current input set is not cleanly 2D. It is closer to a 3D setup.

### 2.2 `summary.csv`

This file contains 160 prediction results:

```text
40 cases × 4 models = 160 results
```

Models included:

```text
PINN
FNO
MeshGraphNet
Transformer-style baseline
```

Materials included:

```text
basalt
sandstone
```

All model calls have status:

```text
ok
```

Important observation:

```text
FNO uses effective_domain_type = rect_2d
domain_adaptation = rect_3d_to_rect_2d
```

while the other models mostly keep:

```text
effective_domain_type = rect_3d
domain_adaptation = none
```

This means the current results are not fully consistent across models, because FNO is effectively evaluated in 2D while the other services receive 3D-style inputs.

---

## 3. Main Decision

For the thesis practical part, the cleanest approach is:

```text
Convert all experimental inputs to 2D.
Run all four models on the same 2D cases.
Generate a new summary_2d.csv.
Create graphs from summary_2d.csv.
Use these results in Chapter 5.
```

This avoids the methodological problem where one model is adapted to 2D and others are still treated as 3D.

---

## 4. Required 2D Input Format

Each case should use:

```json
{
  "requested_domain_type": "rect_2d",
  "input": {
    "source": {
      "z": 0.0,
      "direction": [dx, dy, 0.0]
    },
    "probe": {
      "z": 0.0
    },
    "domain": {
      "type": "rect_2d",
      "size": {
        "lx": 1.0,
        "ly": 1.0,
        "lz": 0.0
      },
      "resolution": {
        "nx": 128,
        "ny": 128,
        "nz": 1
      }
    }
  }
}
```

Required 2D constraints:

```text
source_z = 0.0
probe_z = 0.0
direction_z = 0.0
elevation_deg = 0.0
domain.type = rect_2d
domain.size.lz = 0.0
domain.resolution.nz = 1
```

The direction vector should be normalized in the x-y plane:

```text
d = [dx, dy, 0]
d_hat = d / ||d||
```

---

## 5. Practical Workflow

### Step 1 — Audit the repository

Check the repository:

```text
https://github.com/askarov32/ai-directional-prediction
```

Inspect:

```text
README.md
docker-compose.yml
backend/
pinn-service/
fno-service/
mock-meshgraphnet/
mock-transformer/
frontend/
scripts/
data/
artifacts/
```

Goal:

```text
Understand how each model receives predict requests and how summary.csv is generated.
```

---

### Step 2 — Create clean 2D input cases

Create a new file:

```text
model_comparison_inputs_2d.jsonl
```

Base it on the current:

```text
model_comparison_inputs.jsonl
```

but convert every case to 2D.

Rules:

```text
requested_domain_type: rect_2d
domain.type: rect_2d
lz: 0.0
nz: 1
source.z: 0.0
probe.z: 0.0
source.direction[2]: 0.0
```

Keep the useful scenario variation:

```text
material
temperature_c
pressure_mpa
time_ms
frequency_hz
source_x
source_y
probe_x
probe_y
direction_x
direction_y
boundary_conditions
```

Recommended material set:

```text
basalt
sandstone
```

Optional, if material presets are available and stable:

```text
granite
limestone
```

For a compact and defensible experiment, use:

```text
2 materials × 20 cases × 4 models = 160 predictions
```

or:

```text
4 materials × 10 cases × 4 models = 160 predictions
```

---

### Step 3 — Run predictions for all four models

Run the same 2D cases against:

```text
PINN
FNO
MeshGraphNet
Transformer-style baseline
```

Expected output file:

```text
summary_2d.csv
```

The summary should include:

```text
case_id
model
status
material
temperature_c
pressure_mpa
time_ms
frequency_hz
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
effective_domain_type
domain_adaptation
fallback_used
http_status
error_code
error_message
```

Validation rule:

```text
effective_domain_type should be rect_2d for all models.
direction_z should be 0.0 for all rows.
elevation_deg should be 0.0 or near 0.0.
status should be ok for all successful cases.
```

---

## 6. Sanity Checks Before Graphs

Before plotting, run checks on `summary_2d.csv`.

### 6.1 Domain consistency

Check:

```text
requested_domain_type = rect_2d
effective_domain_type = rect_2d
domain_adaptation = none or consistently documented
```

### 6.2 Geometry consistency

Check:

```text
source_z = 0
probe_z = 0
direction_z = 0
elevation_deg = 0
```

### 6.3 Model status

Check:

```text
status = ok
http_status = 200
fallback_used = false, or explain if true
```

### 6.4 Outlier detection

Check the main outputs:

```text
magnitude
travel_time_ms_pred
max_displacement
max_temperature_perturbation
```

If one model is orders of magnitude larger than the others, do not hide it. Handle it by:

```text
using log-scale plots
plotting that model separately
normalizing values
marking it as unstable/outlier
explaining it in the discussion
```

Current warning from the old `summary.csv`:

```text
FNO outputs appear much larger than other models.
This may indicate scale mismatch, normalization issue, fallback behavior, or incompatible output interpretation.
```

---

## 7. Graphs to Generate

The practical part should not have too many graphs. Use a compact set that directly supports Chapter 5.

### Graph 1 — Travel Time by Model and Material

File:

```text
figures/results/travel_time_by_model_material.png
```

Purpose:

```text
Compare predicted travel time across models and materials.
```

Recommended plot:

```text
bar plot or box plot
x-axis: model
y-axis: travel_time_ms_pred
group/color: material
```

---

### Graph 2 — Maximum Displacement by Model and Material

File:

```text
figures/results/max_displacement_by_model_material.png
```

Purpose:

```text
Compare predicted mechanical response amplitude.
```

Recommended plot:

```text
box plot
x-axis: model
y-axis: max_displacement
group/color: material
```

If values differ strongly by model:

```text
use log scale
```

---

### Graph 3 — Maximum Temperature Perturbation by Model and Material

File:

```text
figures/results/max_temperature_perturbation_by_model_material.png
```

Purpose:

```text
Compare predicted thermal response.
```

Recommended plot:

```text
box plot or bar plot
x-axis: model
y-axis: max_temperature_perturbation
group/color: material
```

If values differ strongly:

```text
use log scale
```

---

### Graph 4 — Basalt vs Sandstone Comparison

File:

```text
figures/results/basalt_vs_sandstone_response.png
```

Purpose:

```text
Show how rock type affects predicted thermoelastic response.
```

Recommended plot:

```text
grouped bar chart
x-axis: material
y-axis: selected output metric
separate panels or separate images for:
- travel time
- displacement
- temperature perturbation
```

---

### Graph 5 — Directional Response by Azimuth

File:

```text
figures/results/directional_response_by_azimuth.png
```

Purpose:

```text
Show how response changes with propagation direction.
```

Recommended plot:

```text
scatter or line plot
x-axis: azimuth_deg
y-axis: selected response metric
group/color: model or material
```

Good metrics:

```text
travel_time_ms_pred
max_displacement
max_temperature_perturbation
```

---

### Graph 6 — Model Stability / Spread

File:

```text
figures/results/model_stability_spread.png
```

Purpose:

```text
Compare how stable or variable the model outputs are across cases.
```

Recommended plot:

```text
box plot
x-axis: model
y-axis: normalized output metric
```

Alternative:

```text
coefficient of variation table/plot
```

---

### Optional Graph 7 — Heatmap

File:

```text
figures/results/response_heatmap_material_model.png
```

Purpose:

```text
Provide a compact visual matrix of model response by material and metric.
```

Recommended plot:

```text
heatmap
rows: model
columns: material or metric
values: normalized mean response
```

Use this only if it is easy and readable.

---

## 8. Result Tables to Generate

Create summary tables for Chapter 5.

### Table 1 — Experiment Overview

```text
number of cases
number of materials
number of models
domain type
main input variables
main output variables
```

### Table 2 — Mean Results by Model and Material

Columns:

```text
model
material
n_cases
mean_travel_time_ms
mean_max_displacement
mean_max_temperature_perturbation
std_travel_time_ms
std_max_displacement
std_max_temperature_perturbation
```

### Table 3 — Data Quality / Status

Columns:

```text
model
n_total
n_ok
n_error
fallback_count
effective_domain_type
notes
```

---

## 9. Chapter 4 Draft Structure

Chapter 4 should describe implementation, not theory.

Recommended structure:

```text
4.1 System Overview
4.2 Repository and Service Architecture
4.3 Prediction API Contract
4.4 Geological Material Presets and Input Data
4.5 2D Experimental Setup
4.6 Model Services
4.6.1 PINN Service
4.6.2 FNO Service
4.6.3 MeshGraphNet Service
4.6.4 Transformer-style Baseline
4.7 Experiment Execution Pipeline
4.8 Summary of Chapter 4
```

Key statement:

```text
The implementation evaluates the directional prediction framework on simplified 2D cases, while the theoretical formulation remains general in 3D.
```

---

## 10. Chapter 5 Draft Structure

Chapter 5 should present results and discussion.

Recommended structure:

```text
5.1 Experiment Setup
5.2 Data Quality and Model Availability
5.3 Overall Model Comparison
5.4 Material-Based Comparison: Basalt and Sandstone
5.5 Directional Response Analysis
5.6 Stability and Outlier Analysis
5.7 Discussion of Physical Consistency
5.8 Limitations of the Results
5.9 Summary of Chapter 5
```

Important wording:

```text
The results are interpreted as comparative directional predictions produced by a research prototype.
They are not treated as fully validated field-scale geophysical simulations.
```

---

## 11. Final Practical Deliverables

The practical part should produce these files:

```text
model_comparison_inputs_2d.jsonl
summary_2d.csv
figures/results/travel_time_by_model_material.png
figures/results/max_displacement_by_model_material.png
figures/results/max_temperature_perturbation_by_model_material.png
figures/results/basalt_vs_sandstone_response.png
figures/results/directional_response_by_azimuth.png
figures/results/model_stability_spread.png
figures/results/response_heatmap_material_model.png
tables/results/model_material_summary.csv
tables/results/model_status_summary.csv
```

Optional LaTeX exports:

```text
tables/results/model_material_summary.tex
tables/results/model_status_summary.tex
```

---

## 12. Short Action Plan

### Immediate next steps

```text
1. Convert current 3D input cases into clean 2D cases.
2. Save them as model_comparison_inputs_2d.jsonl.
3. Run all four model services on the 2D cases.
4. Save results as summary_2d.csv.
5. Validate that all results are truly 2D.
6. Generate result graphs.
7. Write Chapter 4 based on repository architecture and experiment pipeline.
8. Write Chapter 5 based on summary_2d.csv and generated graphs.
```

### If time is limited

Minimum acceptable practical result:

```text
1. Use basalt and sandstone only.
2. Use 20 cases per material.
3. Run all 4 models.
4. Generate 5 core graphs.
5. Discuss results as comparative prototype outputs.
```

---

## 13. Safe Thesis Wording

Use this wording in the thesis:

```text
The theoretical formulation in Chapter 2 is written in three-dimensional form because real geological media are spatially three-dimensional. However, the practical experiments in this thesis are performed on simplified two-dimensional cross-sections and selected source–probe propagation paths. This reduction is used to keep the prototype computationally feasible and to ensure consistent comparison between model services. Therefore, the reported outputs should be interpreted as comparative directional predictions of a research prototype, not as fully validated three-dimensional field-scale simulations.
```


---

# Part B — Codex Task for Generating 2D Result Graphs

# Codex Task: Generate 2D Thesis Result Graphs for Thermoelastic Wave Prediction

You need to generate the practical result graphs for my bachelor thesis.

Thesis title:

```text
AI Directional Prediction of Thermoelastic Wave Propagation in Geological Media
```

The graphs must be consistent with the thesis theory, methodology, and practical 2D implementation.

---

## 1. Thesis Context

The thesis already has:

```text
Chapter 1 — Physical and Geological Foundations of Thermoelastic Wave Propagation
Chapter 2 — Mathematical Formulation of the Thermoelastic Wave Propagation Problem
Chapter 3 — Methodology of Directional Thermoelastic Wave Prediction
Chapter 4 — Implementation and System Architecture
Chapter 5 — Results and Discussion
```

The theoretical formulation in Chapter 2 is written in general 3D form because real geological media are three-dimensional.

However, the practical implementation and results must be treated as:

```text
2D experimental setup
2D cross-section
selected source–probe propagation directions
```

Do **not** present the results as a full validated 3D field-scale thermoelastic simulation.

Correct thesis wording:

```text
The theoretical formulation is general in 3D, while the practical experiments are performed on simplified 2D domains and selected source–probe propagation paths.
```

The result graphs should support this idea:

```text
physical material parameters → model prediction outputs → comparative interpretation
```

---

## 2. Repository Context

Use the repository:

```text
https://github.com/askarov32/ai-directional-prediction
```

Inspect the current project structure before editing.

Important areas to inspect:

```text
README.md
docker-compose.yml
backend/
frontend/
pinn-service/
fno-service/
mgn-service/ or mock-meshgraphnet/
transformer-service/ or mock-transformer/
backend/data/media/catalog.json
scripts/
data/
artifacts/
chapters/
figures/
tables/
```

If folder names differ, adapt to the actual repository.

---

## 3. Input Data

Use the current 2D prediction summary file if it exists:

```text
summary_2d.csv
```

If it does not exist yet, use the available summary CSV and clearly warn that the data may still include non-2D cases.

Use the combined geological material parameter table:

```text
combined_geological_media_parameters.csv
```

or equivalent file generated earlier.

The final graph dataset should be produced by joining:

```text
summary_2d.csv
+
combined_geological_media_parameters.csv
```

Join key:

```text
material
```

If material names differ in capitalization or naming style, normalize them safely:

```text
basalt
sandstone
granite
limestone
```

Optionally include other materials if they are present and have suitable parameters:

```text
gabbro
diabase
granodiorite
marble
schist
quartzite
```

Do not invent missing values.

---

## 4. Required 2D Consistency Checks

Before generating graphs, validate that the result data is actually 2D.

Check these columns if they exist:

```text
requested_domain_type
effective_domain_type
domain_type
source_z
probe_z
direction_z
elevation_deg
domain_lz
domain_nz
```

The required 2D conditions are:

```text
requested_domain_type = rect_2d
effective_domain_type = rect_2d
source_z = 0.0
probe_z = 0.0
direction_z = 0.0
elevation_deg = 0.0
domain_lz = 0.0
domain_nz = 1
```

If some columns do not exist, do not crash. Report that the check could not be fully performed.

If non-2D rows are found:

- save a warning report;
- exclude non-2D rows from final 2D graphs if possible;
- if exclusion would remove too much data, generate the graphs but add a clear warning.

Create a validation report:

```text
tables/results/data_validation_report.txt
```

The report should include:

```text
number of total rows
number of rows used for graphs
number of excluded rows
number of models
number of materials
whether all used rows are 2D
fallback count per model
status count per model
warnings
```

---

## 5. Physical Parameters to Use

From the geological material table, use these physical parameters when available:

```text
rho_kg_m3
Vp_m_s
Vs_m_s
E_Pa
E_GPa
K_Pa
K_GPa
mu_Pa
mu_GPa
k_W_mK
Cp_J_kgK
alpha_1_K
porosity_percent
```

If only Pa values exist, create GPa helper columns for plotting:

```text
E_GPa = E_Pa / 1e9
K_GPa = K_Pa / 1e9
mu_GPa = mu_Pa / 1e9
```

If `alpha_1_K` is very small, create:

```text
alpha_1e6_K = alpha_1_K * 1e6
```

for plotting in units of:

```text
10^-6 K^-1
```

If `porosity_percent` includes strings such as:

```text
total 31.5; eff. 26.5
```

extract the total porosity if safe. If not safe, leave as missing and warn.

---

## 6. Model Prediction Outputs to Use

From `summary_2d.csv`, use these model outputs if available:

```text
travel_time_ms_pred
max_displacement
max_temperature_perturbation
magnitude
azimuth_deg
elevation_deg
status
fallback_used
effective_domain_type
domain_adaptation
model
material
case_id
```

Use only rows where:

```text
status = ok
```

unless there is a specific reason to include failed rows in a status summary.

Do not hide fallback mode. If `fallback_used` exists, include fallback counts in the report.

---

## 7. Main Required Graphs

Generate the following graphs.

All graph files should be saved into:

```text
figures/results/
```

Use high-resolution PNG:

```text
dpi = 300
```

Also save SVG if easy:

```text
figures/results/svg/
```

Use clean academic style:

```text
readable labels
consistent model names
legend
grid
no excessive decoration
units in axis labels
```

Do not use 3D plots.

### Graph 1 — P-wave Velocity vs Predicted Travel Time

File:

```text
figures/results/vp_vs_travel_time_by_model.png
```

Type:

```text
scatter plot
```

Axes:

```text
x-axis: Vp_m_s
y-axis: travel_time_ms_pred
color/hue: model
marker/style: material
```

Physical interpretation:

```text
Higher P-wave velocity should generally correspond to shorter predicted travel time for comparable source–probe distances.
```

This is one of the most important graphs.

### Graph 2 — Density vs Maximum Displacement

File:

```text
figures/results/density_vs_max_displacement_by_model.png
```

Type:

```text
scatter plot
```

Axes:

```text
x-axis: rho_kg_m3
y-axis: max_displacement
color/hue: model
marker/style: material
```

Physical interpretation:

```text
Higher density increases inertia and may reduce displacement response under comparable excitation, although stiffness also matters.
```

If values differ by orders of magnitude, use log scale for y-axis and note this.

### Graph 3 — Young’s Modulus vs Maximum Displacement

File:

```text
figures/results/young_modulus_vs_max_displacement_by_model.png
```

Type:

```text
scatter plot
```

Axes:

```text
x-axis: E_GPa
y-axis: max_displacement
color/hue: model
marker/style: material
```

Physical interpretation:

```text
Stiffer rocks may show smaller deformation under comparable thermal-mechanical excitation.
```

Use `E_Pa / 1e9` if `E_GPa` does not already exist.

### Graph 4 — Thermal Conductivity vs Maximum Temperature Perturbation

File:

```text
figures/results/thermal_conductivity_vs_temperature_perturbation_by_model.png
```

Type:

```text
scatter plot
```

Axes:

```text
x-axis: k_W_mK
y-axis: max_temperature_perturbation
color/hue: model
marker/style: material
```

Physical interpretation:

```text
Thermal conductivity controls heat redistribution and can affect local temperature perturbation.
```

If values differ by orders of magnitude, use log scale for y-axis.

### Graph 5 — Thermal Expansion Coefficient vs Response Magnitude

File:

```text
figures/results/thermal_expansion_vs_response_magnitude_by_model.png
```

Type:

```text
scatter plot
```

Axes:

```text
x-axis: alpha_1e6_K
y-axis: magnitude
color/hue: model
marker/style: material
```

Physical interpretation:

```text
Thermal expansion coefficient links temperature change with thermal strain.
```

Use only rows where thermal expansion coefficient is available.

If `magnitude` is missing, use `max_displacement` instead and name the file:

```text
thermal_expansion_vs_max_displacement_by_model.png
```

### Graph 6 — Travel Time by Material and Model

File:

```text
figures/results/travel_time_by_material_and_model.png
```

Type:

```text
grouped bar plot or box plot
```

Axes:

```text
x-axis: material
y-axis: travel_time_ms_pred
group/hue: model
```

Use mean values with error bars if possible.

Error bars:

```text
standard deviation or standard error
```

Purpose:

```text
Simple comparison of predicted travel time across rock types and models.
```

### Graph 7 — Maximum Displacement by Material and Model

File:

```text
figures/results/max_displacement_by_material_and_model.png
```

Type:

```text
grouped bar plot or box plot
```

Axes:

```text
x-axis: material
y-axis: max_displacement
group/hue: model
```

Use log scale if necessary.

Purpose:

```text
Compare predicted mechanical response across materials and models.
```

### Graph 8 — Directional Response by Azimuth

File:

```text
figures/results/directional_response_by_azimuth.png
```

Type:

```text
scatter plot or line plot
```

Axes:

```text
x-axis: azimuth_deg
y-axis: travel_time_ms_pred
color/hue: model
marker/style: material
```

Purpose:

```text
Show how model response changes with propagation direction in the 2D setup.
```

2D requirement:

```text
elevation_deg should be 0 or near 0.
direction_z should be 0.
```

---

## 8. Optional Graphs

Generate these only if the data is suitable and time allows.

### Optional Graph 9 — Model Stability / Output Spread

File:

```text
figures/results/model_stability_boxplot.png
```

Type:

```text
box plot
```

Axes:

```text
x-axis: model
y-axis: magnitude or max_displacement
```

Purpose:

```text
Identify unstable models or large output spread.
```

Use log scale if needed.

### Optional Graph 10 — Physical Parameter / Output Correlation Heatmap

File:

```text
figures/results/physical_parameters_response_correlation_heatmap.png
```

Type:

```text
heatmap
```

Rows/columns should include physical parameters and model outputs:

Physical parameters:

```text
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

Outputs:

```text
travel_time_ms_pred
max_displacement
max_temperature_perturbation
magnitude
```

Purpose:

```text
Summarize how physical material properties relate to predicted outputs.
```

---

## 9. Summary Tables to Generate

Create these output tables in:

```text
tables/results/
```

### Table 1 — Model and Material Summary

File:

```text
tables/results/model_material_summary.csv
```

Columns:

```text
model
material
n_cases
mean_travel_time_ms
std_travel_time_ms
mean_max_displacement
std_max_displacement
mean_max_temperature_perturbation
std_max_temperature_perturbation
mean_magnitude
std_magnitude
```

Also export LaTeX:

```text
tables/results/model_material_summary.tex
```

### Table 2 — Model Status Summary

File:

```text
tables/results/model_status_summary.csv
```

Columns:

```text
model
n_total
n_ok
n_error
fallback_count
effective_domain_types
notes
```

Also export LaTeX:

```text
tables/results/model_status_summary.tex
```

### Table 3 — Graph Dataset

File:

```text
tables/results/graph_dataset_2d.csv
```

This should be the cleaned joined dataset used for plotting.

---

## 10. Figure Captions for Thesis

Create a file:

```text
tables/results/figure_captions.md
```

For each graph, write:

```text
figure file name
suggested LaTeX caption
short interpretation paragraph
```

Captions should be academic and cautious.

Do not overclaim.

Correct wording example:

```text
The plot suggests a qualitative relationship between P-wave velocity and predicted travel time under the selected 2D source–probe conditions.
```

Incorrect wording:

```text
The model proves the exact physical law of thermoelastic wave propagation.
```

---

## 11. Required Plotting Script

Create a reproducible plotting script:

```text
scripts/generate_2d_result_graphs.py
```

The script should:

```text
1. Load summary_2d.csv.
2. Load combined_geological_media_parameters.csv.
3. Normalize material names.
4. Join the tables by material.
5. Validate 2D consistency.
6. Filter valid rows.
7. Create helper columns E_GPa, K_GPa, mu_GPa, alpha_1e6_K if needed.
8. Generate required graphs.
9. Generate summary tables.
10. Generate figure captions.
11. Save a validation report.
```

The script should be runnable as:

```bash
python scripts/generate_2d_result_graphs.py
```

Also allow optional arguments if easy:

```bash
python scripts/generate_2d_result_graphs.py \
  --summary summary_2d.csv \
  --materials combined_geological_media_parameters.csv \
  --out-figures figures/results \
  --out-tables tables/results
```

Use Python.

Preferred libraries:

```text
pandas
numpy
matplotlib
seaborn
```

If seaborn is already used in the project, it is acceptable. Otherwise, matplotlib alone is fine.

The script must not require internet access.

---

## 12. Robustness Requirements

The script should not fail if optional columns are missing.

Examples:

- if `alpha_1_K` is missing, skip Graph 5 and warn;
- if `Vp_m_s` is missing, skip Graph 1 and warn;
- if `azimuth_deg` is missing, compute it from source/probe or direction if possible;
- if `fallback_used` is missing, report `not available`;
- if `effective_domain_type` is missing, report that domain validation is partial.

All skipped graphs must be listed in:

```text
tables/results/data_validation_report.txt
```

---

## 13. Important Interpretation Rules

The graphs must be interpreted as comparative prototype results.

Use this idea consistently:

```text
The model outputs are comparative directional predictions produced by a research prototype. They are not field-validated measurements and should not be interpreted as exact geophysical simulations.
```

Important model distinction:

```text
PINN is the only explicitly physics-informed component because it uses thermoelastic residuals from the governing equations.
FNO, MeshGraphNet, and Transformer-style services use physically meaningful inputs and outputs, but do not enforce the thermoelastic PDE system in the same way.
```

Do not hide this distinction.

If some model outputs are outliers, do not remove them silently. Either:

```text
plot with log scale
mark in notes
or explain in validation report
```

---

## 14. Expected Final Output

After completing the task, report:

```text
1. Which input files were used.
2. Number of rows loaded.
3. Number of rows used after filtering.
4. Materials included.
5. Models included.
6. Whether all plotted rows are 2D.
7. Which graphs were generated.
8. Which graphs were skipped and why.
9. Which summary tables were generated.
10. Any warnings about fallback mode, outliers, missing values, or non-2D data.
```

Expected generated files:

```text
scripts/generate_2d_result_graphs.py

figures/results/vp_vs_travel_time_by_model.png
figures/results/density_vs_max_displacement_by_model.png
figures/results/young_modulus_vs_max_displacement_by_model.png
figures/results/thermal_conductivity_vs_temperature_perturbation_by_model.png
figures/results/thermal_expansion_vs_response_magnitude_by_model.png
figures/results/travel_time_by_material_and_model.png
figures/results/max_displacement_by_material_and_model.png
figures/results/directional_response_by_azimuth.png

tables/results/graph_dataset_2d.csv
tables/results/model_material_summary.csv
tables/results/model_material_summary.tex
tables/results/model_status_summary.csv
tables/results/model_status_summary.tex
tables/results/data_validation_report.txt
tables/results/figure_captions.md
```

Optional files:

```text
figures/results/model_stability_boxplot.png
figures/results/physical_parameters_response_correlation_heatmap.png
figures/results/svg/*.svg
```

---

## 15. Final Reminder

The purpose of these graphs is not just to compare neural networks.

The purpose is to show:

```text
how physical properties of geological materials relate to predicted thermoelastic response,
and how four model services behave under the same 2D physically motivated input-output framework.
```

