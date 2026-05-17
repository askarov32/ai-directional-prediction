# Prompt for Codex: Improve the PINN Architecture for Thermoelastic Wave Prediction

You are a senior ML engineer, scientific software architect, and PINN reviewer.

You are working with this GitHub repository:

```text
https://github.com/askarov32/ai-directional-prediction
```

The project is a bachelor thesis MVP / research prototype titled:

```text
AI Directional Prediction of Thermoelastic Wave Propagation in Geological Media
```

The repository contains a FastAPI backend, frontend visualization, several model services, and a `pinn-service` used as a physics-informed baseline for predicting thermoelastic wave propagation in geological media.

Your task is to improve the PINN implementation in a scientifically reasonable and technically safe way.

---

## 1. Important scientific positioning

Do **not** present this project as a fully validated real-world geophysical simulator.

Do **not** claim that the PINN fully solves the complete thermoelastic wave propagation problem.

Correct positioning:

- the project is an AI-assisted comparative framework;
- the PINN is a physics-informed neural baseline;
- the model predicts thermoelastic fields under simplified assumptions;
- the current implementation is closer to a research prototype than a production-grade physical simulator;
- physical constraints should be improved, documented, and evaluated, but not overclaimed.

---

## 2. Current PINN architecture to inspect

First, inspect the current `pinn-service` implementation carefully.

Pay special attention to:

```text
pinn-service/src/pinn_service/model.py
pinn-service/src/pinn_service/losses.py
pinn-service/src/pinn_service/training.py
pinn-service/scripts/run_training_experiment.py
pinn-service/README.md
docs/pinn_architecture_details.md
```

The current model is approximately a plain MLP:

```text
Input:  [x, y, z, t, E, nu, rho, alpha, k, Cp]
Output: [T, u, v, w]
Hidden: repeated Linear layers with the same width, e.g. 192
Activation: tanh
```

This is acceptable as a baseline, but it is too simple for a coupled thermoelastic task.

Main issues to address:

1. Coordinates and material parameters are mixed immediately in one vector.
2. Temperature and displacement are predicted through one shared output layer.
3. There are no residual / skip connections.
4. The architecture has uniform hidden width without a clear reason.
5. Boundary and initial conditions are not clearly enforced as separate losses unless already implemented elsewhere.
6. The model should remain API-compatible with the existing backend and frontend.

---

## 3. Required architectural improvement

Implement an improved PINN architecture while keeping the old architecture available as a baseline.

Do **not** delete the current model unless necessary. Prefer adding a new architecture class and a configuration switch.

### 3.1 Add an improved model variant

Create a new model variant, for example:

```text
ResSplitPINN
```

or another clear name.

The model should use this conceptual structure:

```text
Input:
  coords:   [x, y, z, t]
  material: [E, nu, rho, alpha, k, Cp]

Coordinate encoder:
  optional Fourier / positional encoding for x, y, z, t
  Linear -> activation

Material encoder:
  Linear 6 -> 64 -> 64

Fusion:
  concat(coord_features, material_features)

Shared physics trunk:
  residual MLP blocks
  width 192 or 256
  smooth activation

Separate output heads:
  temperature head:   trunk -> hidden -> 1       # T
  displacement head:  trunk -> hidden -> 3       # u, v, w

Final output:
  concat(T, u, v, w)
```

The output order must remain:

```text
[T, u, v, w]
```

The input order must remain:

```text
[x, y, z, t, E, nu, rho, alpha, k, Cp]
```

### 3.2 Use smooth activations

Use activations suitable for PINN-style residual losses:

Preferred:

```text
tanh
silu
gelU
```

Avoid:

```text
ReLU
Dropout
BatchNorm
```

Reason: PDE residuals require stable derivatives. ReLU produces piecewise-constant second derivatives, Dropout makes residuals noisy, and BatchNorm can interfere with coordinate-dependent derivatives.

### 3.3 Add residual blocks

Implement a clean residual block, for example:

```text
x -> Linear -> activation -> Linear -> add skip -> activation
```

Use layer sizes consistently, for example width 192 or 256.

The improved model should not simply add more depth. It should improve gradient flow and feature reuse.

### 3.4 Separate heads for physical fields

Do not use only one final `Linear(hidden, 4)` for all outputs in the improved model.

Use at least two heads:

```text
T_head: predicts temperature perturbation / temperature field
U_head: predicts displacement components u, v, w
```

This is important because thermal and mechanical fields have different physical behavior and different scales.

---

## 4. Optional coordinate encoding

Add optional Fourier / positional encoding for the coordinate part only:

```text
[x, y, z, t]
```

Do not apply Fourier encoding to material parameters.

Make it configurable:

```text
use_fourier_features: true/false
fourier_num_frequencies: int
fourier_scale: float
```

Default can be disabled if you want a safer first implementation.

Important: keep the code simple and well-documented.

---

## 5. Loss function review

Inspect the current PINN loss implementation.

Current expected losses may include something like:

```text
L_total = lambda_sup * L_sup
        + lambda_vel * L_vel
        + lambda_wave * L_wave
        + lambda_temp * L_temp
```

or similar.

Your task:

1. Do not remove existing losses.
2. Make sure the improved model is compatible with all existing losses.
3. Check whether boundary condition and initial condition losses exist explicitly.
4. If they do not exist, do not fake them. Instead:
   - add clear TODO / documentation;
   - optionally add a safe implementation only if the training data already contains boundary / initial condition masks or values;
   - expose configurable loss weights only when the data pipeline supports them.

Do **not** claim that boundary conditions are strictly enforced unless there is actual BC/IC loss code and data support.

---

## 6. Configuration requirements

Add a config mechanism so the user can choose the architecture.

Example:

```text
--architecture mlp
--architecture res_split
```

or via config file / environment variable if the project already uses that style.

Suggested model variants:

```text
mlp          # existing baseline
res_split    # improved ResMLP with separate heads
```

Add training options where appropriate:

```text
--hidden-dim 192
--num-blocks 4
--activation tanh
--use-fourier-features
--fourier-num-frequencies 6
--fourier-scale 1.0
```

Keep backward compatibility with existing scripts.

Existing commands should not break.

---

## 7. Training and evaluation requirements

Add or update experiment support so that the following variants can be compared:

```text
1. Baseline MLP: current 192 x 6 tanh architecture
2. Wider MLP: 256 x 6 tanh
3. Tapered MLP: 256 -> 256 -> 192 -> 192 -> 128 -> 128
4. ResSplitPINN: residual trunk + separate T/u-v-w heads
5. ResSplitPINN + Fourier coordinate features, if implemented safely
```

For each experiment, log at least:

```text
train supervised loss
validation supervised loss
physics residual loss, if available
thermal residual loss, if available
velocity consistency loss, if available
total loss
training time
number of parameters
```

If the project already has a logging format, reuse it.

Do not introduce heavy external dependencies unless necessary.

---

## 8. Documentation requirements

Update documentation after code changes.

Update or create:

```text
pinn-service/README.md
docs/pinn_architecture_details.md
```

The documentation must explain:

1. What the old architecture does.
2. Why the old architecture is kept as a baseline.
3. What the improved architecture changes.
4. Why coordinates and material parameters are encoded separately.
5. Why temperature and displacement use separate heads.
6. Why residual blocks are useful for PINN training.
7. Whether boundary and initial conditions are explicitly enforced or only represented indirectly through data.
8. How to run baseline and improved training experiments.
9. How to compare results.

Use careful scientific wording.

Do not overstate the model quality.

---

## 9. Testing requirements

Add or update tests where appropriate.

At minimum, test:

1. Baseline model still works.
2. Improved model accepts input shape `[batch_size, 10]`.
3. Improved model returns output shape `[batch_size, 4]`.
4. Output order is `[T, u, v, w]`.
5. Training script can instantiate both architectures.
6. Loss functions work with both architectures.
7. Optional Fourier features do not change the public input contract.

If the repository already has a test framework, use it.

Do not introduce unnecessary testing frameworks.

---

## 10. Code quality requirements

Follow the existing repository style.

Keep changes minimal, readable, and maintainable.

Avoid large rewrites unless required.

Use clear class names and type hints.

Add comments only where they explain non-obvious scientific or architectural decisions.

Do not add dead code.

Do not leave broken imports.

Do not hardcode local absolute paths.

Keep Docker compatibility.

---

## 11. Expected final output from Codex

After implementation, provide a concise technical report with:

```text
1. Files changed
2. New classes / functions added
3. How to run the baseline model
4. How to run the improved model
5. How to compare experiments
6. What was intentionally not changed
7. Any limitations or TODOs
```

Also include example commands.

Example format:

```bash
python pinn-service/scripts/run_training_experiment.py \
  --architecture mlp \
  --hidden-dim 192 \
  --depth 6

python pinn-service/scripts/run_training_experiment.py \
  --architecture res_split \
  --hidden-dim 256 \
  --num-blocks 4 \
  --activation tanh
```

Adjust these commands to the real script arguments after inspecting the repository.

---

## 12. Acceptance criteria

The task is complete only if:

- the old PINN baseline still works;
- the improved architecture can be selected through config or CLI;
- the improved model keeps the same input and output API contract;
- training does not crash on a dry run;
- inference still returns `[T, u, v, w]`;
- documentation clearly explains the architecture change;
- no unsupported claim is made about strict physical correctness;
- tests or shape checks are added for the new architecture.

