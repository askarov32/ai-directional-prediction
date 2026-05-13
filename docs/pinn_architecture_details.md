# PINN Architecture Details

This document describes the actual PINN architecture currently implemented in this repository.

It is not a generic PINN description. It documents the concrete network, inputs, outputs, loss terms, training configuration, and inference flow used by the current `pinn-service`.

## 1. Service Location

The PINN implementation lives in:

```text
pinn-service/src/pinn_service/
```

Main files:

- `model.py`: neural network architecture;
- `physics.py`: autograd derivatives and thermoelastic residual utilities;
- `losses.py`: hybrid supervised + physics-informed loss;
- `trainer.py`: training loop, validation, checkpoints, scheduler, metrics;
- `training_data.py`: dataset loading, target selection, scaling;
- `service_app.py`: FastAPI inference service;
- `inference_service.py`: checkpoint loading and prediction postprocessing.

## 2. Network Type

The current PINN model is a fully connected multilayer perceptron:

```text
MLP_PINN
```

Implementation:

```text
pinn-service/src/pinn_service/model.py
```

The model is defined as `torch.nn.Sequential`.

Default architecture:

```text
Input Linear:   10 -> 192
Activation:     Tanh

Hidden block 1: 192 -> 192
Activation:     Tanh

Hidden block 2: 192 -> 192
Activation:     Tanh

Hidden block 3: 192 -> 192
Activation:     Tanh

Hidden block 4: 192 -> 192
Activation:     Tanh

Hidden block 5: 192 -> 192
Activation:     Tanh

Output Linear:  192 -> 4
```

In config terms:

```text
input_dim = 10
output_dim = 4
hidden_dim = 192
depth = 6
activation = tanh
```

Important detail: `depth = 6` means six hidden `Linear` layers total:

- one input-to-hidden layer;
- five hidden-to-hidden layers;
- one final output layer.

Supported activations:

```text
tanh
silu
gelu
relu
```

Default activation:

```text
tanh
```

## 3. Network Inputs

The model input vector has 10 features:

```latex
X = [x, y, z, t, E, \nu, \rho, \alpha, k, C_p]
```

Concrete feature order:

```text
0: x
1: y
2: z
3: t
4: youngs_modulus
5: poissons_ratio
6: density
7: thermal_expansion
8: thermal_conductivity
9: heat_capacity
```

Meaning:

- `x`, `y`, `z`: spatial coordinates;
- `t`: time;
- `youngs_modulus`: Young's modulus `E`;
- `poissons_ratio`: Poisson ratio `nu`;
- `density`: material density `rho`;
- `thermal_expansion`: thermal expansion coefficient `alpha`;
- `thermal_conductivity`: thermal conductivity `k`;
- `heat_capacity`: heat capacity `Cp`.

Inputs are standardized before training:

```latex
X_{scaled} = \frac{X - \mu_X}{\sigma_X}
```

The scaler is saved in:

```text
scalers.json
```

## 4. Network Outputs

The primary network output has 4 values:

```latex
\hat{Y} = [\hat{T}, \hat{u}, \hat{v}, \hat{w}]
```

Concrete output order:

```text
0: temperature_k
1: disp_x
2: disp_y
3: disp_z
```

Meaning:

- `temperature_k`: predicted temperature in Kelvin;
- `disp_x`: displacement along x;
- `disp_y`: displacement along y;
- `disp_z`: displacement along z.

Outputs are also standardized for supervised training:

```latex
Y_{scaled} = \frac{Y - \mu_Y}{\sigma_Y}
```

During physics residual calculation, model outputs are converted back to physical units.

## 5. Additional Training Targets

The dataset also contains velocity targets:

```text
vel_x
vel_y
vel_z
```

The neural network does not directly output velocity. Velocity is computed through autograd:

```latex
\hat{u}_t = \frac{\partial \hat{u}}{\partial t}
```

```latex
\hat{v}_t = \frac{\partial \hat{v}}{\partial t}
```

```latex
\hat{w}_t = \frac{\partial \hat{w}}{\partial t}
```

Then these derivatives are compared against the dataset velocity targets.

## 6. Physics Mode

Current default physics mode:

```text
coupled_thermoelastic
```

Compatibility mode:

```text
simple_heat
```

The experiment runner always uses:

```text
physics_mode = coupled_thermoelastic
```

## 7. Material Assumption

The current MVP treats material parameters as pointwise locally homogeneous features.

That means:

- `E`, `nu`, `rho`, `alpha`, `k`, and `Cp` are input features;
- they are unscaled before computing PDE residuals;
- the model does not take spatial derivatives of material parameters.

This is intentional for the current dataset format because material parameters are provided as features, not as differentiable spatial fields.

## 8. Loss Function

The total training objective is:

```latex
\mathcal{L}_{total}
=
\lambda_{sup}\mathcal{L}_{sup}
+
\lambda_{vel}\mathcal{L}_{vel}
+
\lambda_{wave}\mathcal{L}_{wave}
+
\lambda_{temp}\mathcal{L}_{temp}
```

Default weights:

```text
supervised_weight = 1.0
velocity_weight = 0.25
wave_residual_weight = 0.1
thermal_residual_weight = 0.05
```

### Supervised Loss

The supervised loss compares predicted primary fields against COMSOL-derived targets:

```latex
\mathcal{L}_{sup}
=
\operatorname{MSE}
\left(
[\hat{T}, \hat{u}, \hat{v}, \hat{w}],
[T, u, v, w]
\right)
```

### Velocity Consistency Loss

Velocity is computed from displacement derivatives:

```latex
\mathcal{L}_{vel}
=
\operatorname{MSE}
\left(
\left[
\frac{\partial \hat{u}}{\partial t},
\frac{\partial \hat{v}}{\partial t},
\frac{\partial \hat{w}}{\partial t}
\right],
[u_t, v_t, w_t]
\right)
```

### Elastic Wave Residual Loss

The elastic wave residual is:

```latex
R_i^{wave}
=
\rho
\frac{\partial^2 u_i}{\partial t^2}
-
\frac{\partial \sigma_{ij}}{\partial x_j}
```

The wave loss is:

```latex
\mathcal{L}_{wave}
=
\operatorname{mean}(R_u^2 + R_v^2 + R_w^2)
```

### Coupled Thermal Residual Loss

The coupled thermal residual is:

```latex
R_T
=
\rho C_p
\frac{\partial T}{\partial t}
-
k\nabla^2T
+
\gamma T_0
\frac{\partial \varepsilon_{kk}}{\partial t}
```

The thermal loss is:

```latex
\mathcal{L}_{temp}
=
\operatorname{mean}(R_T^2)
```

Default reference temperature:

```text
T0 = 293.15 K
```

## 9. Thermoelastic Physics Components

Lame parameters:

```latex
\mu = \frac{E}{2(1+\nu)}
```

```latex
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
```

Thermoelastic coupling:

```latex
\gamma = (3\lambda + 2\mu)\alpha
```

Small strain:

```latex
\varepsilon_{ij}
=
\frac{1}{2}
\left(
\frac{\partial u_i}{\partial x_j}
+
\frac{\partial u_j}{\partial x_i}
\right)
```

Volumetric strain:

```latex
\varepsilon_{kk}
=
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
+
\frac{\partial w}{\partial z}
```

Thermoelastic stress:

```latex
\sigma_{ij}
=
\lambda \delta_{ij}\varepsilon_{kk}
+
2\mu\varepsilon_{ij}
-
\gamma \delta_{ij}(T - T_0)
```

## 10. Normalization In Physics Residuals

Although neural inputs and outputs are standardized, PDE residuals are computed in physical units.

Coordinate derivative correction:

```latex
\frac{\partial}{\partial x_{phys}}
=
\frac{1}{\sigma_x}
\frac{\partial}{\partial x_{scaled}}
```

Second derivative correction:

```latex
\frac{\partial^2}{\partial x_{phys}^2}
=
\frac{1}{\sigma_x^2}
\frac{\partial^2}{\partial x_{scaled}^2}
```

The same correction is applied to `y`, `z`, and `t`.

## 11. Training Defaults

The recommended experiment runner is:

```text
pinn-service/scripts/run_training_experiment.py
```

Current default training settings:

```text
epochs = 2000
batch_size = 8192
learning_rate = 5e-4
min_learning_rate = 1e-6
weight_decay = 1e-6
hidden_dim = 192
depth = 6
activation = tanh
loss_balance_mode = normalize
max_grad_norm = 1.0
lr_scheduler_patience = 40
lr_scheduler_factor = 0.5
early_stopping_patience = 250
seed = 42
```

Optimizer:

```text
AdamW
```

Learning-rate scheduler:

```text
ReduceLROnPlateau
```

Checkpoint selection:

```text
best_model.pth is selected by val_total_loss when validation data is available.
```

## 12. Training Data Flow

Raw CSV data:

```text
data/granite/
data/limestone/
data/sandstone/
data/basalt/
```

Dataset build:

```text
build_rod_experiments.py
```

Combined dataset:

```text
pinn-service/artifacts/rod_experiments/training_samples_all_rocks.npz
```

Train/validation split:

```text
create_train_val_split.py
```

Split outputs:

```text
pinn-service/artifacts/rod_experiments/splits/train_samples.npz
pinn-service/artifacts/rod_experiments/splits/val_samples.npz
```

Loss scale estimation:

```text
estimate_loss_scales.py
```

Loss scale output:

```text
pinn-service/artifacts/rod_experiments/reports/loss_scale_report.json
```

Training:

```text
run_training_experiment.py
```

Training outputs:

```text
model.pth
best_model.pth
metrics.json
metrics.csv
training_config.json
scalers.json
report/training_report.html
```

## 13. Inference Architecture

The PINN service exposes:

```text
GET /health
GET /ready
POST /predict
```

Inference steps:

```text
backend enriched prediction request
  -> pinn-service /predict
  -> build feature vector [x,y,z,t,E,nu,rho,alpha,k,Cp]
  -> scale input using checkpoint scaler
  -> run MLP_PINN
  -> unscale [T,u,v,w]
  -> postprocess direction metrics
  -> return frontend-friendly prediction payload
```

The backend then normalizes the PINN response into the shared prediction contract used by the frontend.

## 14. Current Scope

The current PINN is a practical thesis MVP baseline:

- it uses a real neural network checkpoint;
- it trains on COMSOL-derived CSV data;
- it includes supervised data loss;
- it includes velocity consistency;
- it includes elastic wave residual;
- it includes coupled thermal residual;
- it supports validation-based best checkpoint selection;
- it supports loss-scale normalization.

Current limitations:

- no explicit boundary-condition loss yet;
- no explicit initial-condition loss yet;
- no adaptive collocation-point sampler yet;
- material parameters are treated as locally homogeneous pointwise features.
