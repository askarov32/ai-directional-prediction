# Codex Task: Improve Only the FNO Service

## Goal

Improve only the FNO part of the project so that its outputs are numerically stable, physically interpretable, and suitable for comparison in the thesis.

Repository:

```text
https://github.com/askarov32/ai-directional-prediction
```

Thesis title:

```text
AI Directional Prediction of Thermoelastic Wave Propagation in Geological Media
```

The thesis uses a general 3D theoretical formulation, but the practical experiments are performed on simplified 2D domains and selected source–probe directions.

The FNO service must therefore work cleanly with:

```text
domain.type = rect_2d
domain.size.lz = 0.0
domain.resolution.nz = 1
source.z = 0.0
probe.z = 0.0
source.direction[2] = 0.0
direction_z = 0.0
elevation_deg = 0.0
```

---

## Current Problem

The current FNO results appear unstable or incorrectly scaled.

Observed issue from generated result graphs:

```text
FNO produces max_displacement, max_temperature_perturbation, and magnitude values several orders of magnitude larger than PINN, MeshGraphNet, and Transformer-style baseline.
```

This likely indicates one or more of the following:

```text
normalization error
denormalization error
unit mismatch
wrong output channel interpretation
wrong metric calculation
raw tensor norm used as physical displacement
rect_3d_to_rect_2d adaptation issue
training/inference scaling mismatch
```

Do not hide the issue in plots. Diagnose and fix it where possible.

---

## Scope

Work only on the FNO service and files directly required for FNO inference/training/tests/docs.

Do not modify:

```text
PINN service
MeshGraphNet service
Transformer service
frontend
thesis text
general plotting script
```

unless a very small shared schema adjustment is strictly necessary.

Do not remove FNO from the project.

Do not claim that FNO is physics-informed in the same sense as PINN.

Correct positioning:

```text
FNO is a 2D neural operator baseline that predicts physical fields or field-derived quantities.
It uses physically meaningful inputs and outputs, but does not explicitly enforce the thermoelastic PDE residuals like PINN.
```

---

## Step 1: Inspect FNO Files

Inspect the FNO codebase, especially:

```text
fno-service/
fno-service/README.md
fno-service/src/
fno-service/src/fno_service/
fno-service/src/fno_service/models/
fno-service/src/fno_service/api.py
fno-service/src/fno_service/inference*
fno-service/src/fno_service/training*
fno-service/src/fno_service/data*
```

Find and document:

```text
where FNO input tensors are created
where target tensors are loaded
where output channels are defined
where normalization is applied
where normalization statistics are stored
where denormalization is applied
where API prediction response is built
where max_displacement is computed
where max_temperature_perturbation is computed
where magnitude is computed
where rect_3d_to_rect_2d adaptation is performed
```

---

## Step 2: Verify FNO Output Channels

Confirm the actual FNO output channels.

Expected fields are:

```text
temperature or temperature perturbation
disp_x
disp_y
optional disp_z
```

For the thesis 2D experiments:

```text
disp_z should be 0, omitted, or clearly ignored
```

Stress and strain are not direct FNO outputs. Do not add them unless the dataset explicitly supports them.

---

## Step 3: Make FNO Cleanly Support 2D Inputs

FNO must accept clean 2D requests directly:

```text
domain.type = rect_2d
domain.size.lz = 0.0
domain.resolution.nz = 1
source.z = 0.0
probe.z = 0.0
source.direction[2] = 0.0
```

Expected response metadata for clean 2D:

```text
effective_domain_type = rect_2d
domain_adaptation = none
```

If a legacy 3D request is adapted, keep that behavior only if already required, but make it explicit:

```text
domain_adaptation = rect_3d_to_rect_2d
```

For thesis experiments, the preferred path is clean 2D input with no hidden adaptation.

---

## Step 4: Fix or Add Channel-wise Normalization

Temperature and displacement fields have very different scales:

```text
temperature: tens or hundreds
displacement: often very small values
```

FNO must use channel-wise normalization.

Required behavior:

```text
temperature channel normalized separately
disp_x channel normalized separately
disp_y channel normalized separately
disp_z channel normalized separately if used
```

Persist normalization statistics and reuse them consistently for:

```text
training
validation
inference
```

Normalization metadata should include:

```text
channel_names
mean
std
min
max
units
```

Do not use one global scalar normalization across all fields unless there is a very clear reason.

---

## Step 5: Fix Denormalization Before Metrics

If FNO predicts normalized outputs, denormalize them before computing physical metrics.

Correct order:

```text
1. Build normalized input tensor.
2. Run FNO.
3. Obtain normalized predicted output.
4. Denormalize each output channel to physical units.
5. Compute response metrics from denormalized fields.
6. Return physical metrics in API response.
```

Incorrect behavior to eliminate:

```text
max_displacement computed from normalized tensor
magnitude computed from raw output tensor
temperature perturbation computed from latent representation
metrics computed before denormalization
```

---

## Step 6: Fix Metric Calculation

For 2D displacement, compute:

```text
max_displacement = max(sqrt(disp_x^2 + disp_y^2))
```

Do not compute max displacement as:

```text
sum over grid
sum over channels
global tensor norm
loss value
latent vector norm
raw unscaled tensor value
```

For temperature perturbation:

If the model predicts temperature:

```text
theta = T - T0
max_temperature_perturbation = max(abs(theta))
```

If the model predicts theta directly:

```text
max_temperature_perturbation = max(abs(theta_pred))
```

For magnitude:

Make the definition explicit. If it is a comparative score rather than a physical unit, document that in code comments and response metadata.

---

## Step 7: Add FNO Sanity Checks

Before returning a prediction, validate:

```text
no NaN
no Inf
finite outputs
reasonable displacement scale
reasonable temperature perturbation scale
reasonable magnitude scale
```

If outputs are suspicious, include warnings in the response:

```text
warnings = ["scale_outlier"]
```

Depending on project conventions, either keep:

```text
status = ok
```

with warnings, or return:

```text
status = unstable
```

Suggested conservative warning thresholds:

```text
abs(max_displacement) > 1e2
abs(max_temperature_perturbation) > 1e4
magnitude > 1e6
```

These are prototype sanity checks, not universal physical limits. Add comments explaining that.

---

## Step 8: Add FNO Response Metadata

FNO prediction response should include or preserve:

```text
model
model_version
status
effective_domain_type
domain_adaptation
fallback_used
normalization_used
denormalization_used
warnings
```

This is needed so the thesis can honestly discuss FNO behavior.

---

## Step 9: Add or Update FNO Tests

Add tests for FNO only.

Test cases should check:

```text
clean rect_2d input is accepted
effective_domain_type is rect_2d
domain_adaptation is none for clean rect_2d input
source.z, probe.z, and direction_z remain zero
normalization metadata is loaded or handled safely
denormalization is applied before metric calculation
max_displacement uses sqrt(disp_x^2 + disp_y^2)
NaN outputs are detected
Inf outputs are detected
scale outliers produce warnings
response metadata is present
```

Do not require internet access.

---

## Step 10: Update FNO Documentation Only

Update only FNO-specific documentation, for example:

```text
fno-service/README.md
```

Explain:

```text
FNO is a 2D neural operator baseline.
FNO predicts temperature/displacement fields or field-derived quantities.
FNO uses channel-wise normalization.
FNO outputs are denormalized before physical metrics are computed.
FNO does not explicitly enforce thermoelastic PDE residuals like PINN.
FNO should be evaluated using clean rect_2d thesis inputs.
```

Also mention limitations:

```text
FNO results depend on training data, scaling, normalization statistics, and output interpretation.
If warnings appear, they must be discussed as instability or scale mismatch.
```

---

## Step 11: Final Report

After making changes, report:

```text
1. FNO files inspected.
2. Root cause found, or most likely cause if not fully proven.
3. FNO files changed.
4. How normalization works now.
5. How denormalization works now.
6. How max_displacement is computed now.
7. How max_temperature_perturbation is computed now.
8. How clean 2D inputs are handled now.
9. What tests were added.
10. What limitations remain.
11. Whether existing FNO results should be regenerated.
```

---

## Constraints

Do not rewrite the whole project.

Do not modify non-FNO model services.

Do not hide FNO instability.

Do not remove FNO from model comparison.

Do not claim physical validation.

Do not claim FNO solves the full coupled thermoelastic PDE system.

The goal is to make FNO scale-consistent, 2D-consistent, and honestly interpretable as a neural operator baseline for the thesis.
