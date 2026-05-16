# PINN Architecture Details

This document describes the concrete PINN architectures currently implemented in this repository.

It is intentionally scoped to the actual `pinn-service` code, not to an idealized full thermoelastic simulator. The current PINN should be interpreted as a physics-informed neural baseline inside a comparative research prototype.

## 1. Where The PINN Lives

Core implementation:

- `pinn-service/src/pinn_service/model.py`
- `pinn-service/src/pinn_service/physics.py`
- `pinn-service/src/pinn_service/losses.py`
- `pinn-service/src/pinn_service/trainer.py`
- `pinn-service/src/pinn_service/inference_service.py`
- `pinn-service/src/pinn_service/train.py`
- `pinn-service/scripts/run_training_experiment.py`

## 2. Public Input And Output Contract

The public neural contract is unchanged for all PINN variants.

Input feature order:

```latex
X = [x, y, z, t, E, \nu, \rho, \alpha, k, C_p]
```

Concrete order:

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

Output feature order:

```latex
\hat{Y} = [\hat{T}, \hat{u}, \hat{v}, \hat{w}]
```

Concrete order:

```text
0: temperature_k
1: disp_x
2: disp_y
3: disp_z
```

This means the backend, frontend, and inference payload format do not need to change when switching architectures.

## 3. Why The Old Baseline Is Still Kept

The original `MLP_PINN` is kept for three reasons:

1. It is the simplest reference point for ablation.
2. Existing checkpoints remain loadable.
3. It provides a stable baseline when evaluating whether architectural changes help or only add complexity.

The baseline is a plain fully connected MLP:

```text
input -> hidden -> hidden -> ... -> hidden -> output
```

Typical baseline setting:

```text
architecture = mlp
hidden_dim = 192
depth = 6
activation = tanh
```

The baseline can also be widened or tapered by supplying explicit hidden sizes through:

```text
--mlp-layer-dims 256,256,192,192,128,128
```

## 4. Improved Architecture: ResSplitPINN

The improved variant is:

```text
architecture = res_split
```

It keeps the same input and output contract but changes the internal structure:

```text
coords [x,y,z,t] ------> coordinate encoder ----\
                                                 \
                                                  -> fusion -> residual trunk -> T head -> 1
                                                 /
material [E,nu,rho,alpha,k,Cp] -> material encoder -/                      -> U head -> 3
```

### 4.1 Coordinate And Material Separation

Coordinates and material parameters are not mixed immediately anymore.

Why this helps:

- coordinates describe geometry and time;
- material features describe constitutive behavior;
- separating them makes the feature flow easier to interpret;
- it reduces the chance that the network treats all ten inputs as one undifferentiated tabular vector.

### 4.2 Material Encoder

The material branch is a small MLP:

```text
6 -> 64 -> 64
```

with a smooth activation between layers.

### 4.3 Coordinate Encoder

The coordinate branch first receives `[x, y, z, t]`.

If Fourier features are disabled:

```text
4 -> Linear -> activation
```

If Fourier features are enabled:

```text
[x, y, z, t] -> Fourier encoding -> Linear -> activation
```

Fourier encoding is applied only to coordinates, never to material parameters.

### 4.4 Residual Trunk

The fused representation passes through residual MLP blocks of the form:

```text
x -> Linear -> activation -> Linear -> add skip -> activation
```

This is not meant to make the network arbitrarily deep. The main reason is safer optimization and better feature reuse under derivative-based PINN losses.

### 4.5 Separate Physical Heads

The improved model uses two output heads:

- `temperature_head`: predicts `T`
- `displacement_head`: predicts `[u, v, w]`

This is useful because thermal and mechanical fields have different scales and different physical roles. A single final layer can still work, but split heads are a more reasonable inductive bias for a coupled thermoelastic baseline.

## 5. Optional Fourier Coordinate Features

The improved architecture supports optional Fourier features:

```text
--use-fourier-features
--fourier-num-frequencies 6
--fourier-scale 1.0
```

These features only affect the internal coordinate encoding. The external input vector remains the same 10-dimensional contract.

This option is experimental and should be treated as a controlled architectural variant, not as a guaranteed improvement.

## 6. Supported Architecture Variants

The current training stack can compare at least the following variants:

1. Baseline MLP:

```text
--architecture mlp --hidden-dim 192 --depth 6 --activation tanh
```

2. Wider MLP:

```text
--architecture mlp --hidden-dim 256 --depth 6 --activation tanh
```

3. Tapered MLP:

```text
--architecture mlp --mlp-layer-dims 256,256,192,192,128,128 --activation tanh
```

4. ResSplitPINN:

```text
--architecture res_split --hidden-dim 192 --num-blocks 4 --activation tanh
```

5. ResSplitPINN with Fourier coordinate features:

```text
--architecture res_split --hidden-dim 192 --num-blocks 4 --activation tanh --use-fourier-features
```

## 7. Loss Compatibility

Both architectures are trained with the same hybrid loss stack:

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

Where:

- `L_sup` is supervised field loss on `[T,u,v,w]`;
- `L_vel` is velocity consistency loss on `[u_t,v_t,w_t]`;
- `L_wave` is the elastic wave residual loss;
- `L_temp` is the thermal residual loss.

The architecture change does not modify the public loss interface.

Supported physics modes:

- `coupled_thermoelastic`: 3D coupled thermoelastic residual using x, y, z, and t derivatives.
- `simple_heat`: compatibility mode with a heat residual and no wave residual contribution.
- `plane_strain_2d`: strict 2D mode for derived 2D artifacts; it uses x-y derivatives only, zero out-of-plane strain terms, `[u,v]` wave residuals, and an x-y thermal Laplacian.

## 8. What Is Still Not Explicitly Enforced

The current implementation does **not** claim strict enforcement of:

- boundary conditions through dedicated BC masks and BC loss terms;
- initial conditions through dedicated IC masks and IC loss terms.

At the moment, these effects are represented only indirectly through the supervised data and the PDE-style residual terms.

That means this PINN is better described as:

- a hybrid supervised + physics-informed baseline;
- a comparative research model;
- not a fully validated boundary-aware solver.

## 9. Material Modeling Assumption

The current MVP treats:

```text
E, nu, rho, alpha, k, Cp
```

as locally homogeneous pointwise material features.

The model uses them in the residual computation after unscaling, but it does not take their spatial derivatives. This is an intentional simplification tied to the current dataset format.

## 10. How To Run Baseline And Improved Experiments

Baseline MLP:

```bash
PYTHONPATH=pinn-service/src python3 pinn-service/scripts/run_training_experiment.py \
  --output-dir pinn-service/artifacts/checkpoints/pinn_mlp_192x6 \
  --architecture mlp \
  --hidden-dim 192 \
  --depth 6 \
  --activation tanh \
  --epochs 2000 \
  --batch-size 8192 \
  --validation-batch-size 8192 \
  --device cpu
```

Improved ResSplitPINN:

```bash
PYTHONPATH=pinn-service/src python3 pinn-service/scripts/run_training_experiment.py \
  --output-dir pinn-service/artifacts/checkpoints/pinn_res_split \
  --architecture res_split \
  --hidden-dim 192 \
  --num-blocks 4 \
  --activation tanh \
  --epochs 2000 \
  --batch-size 8192 \
  --validation-batch-size 8192 \
  --device cpu
```

Improved ResSplitPINN with Fourier coordinate features:

```bash
PYTHONPATH=pinn-service/src python3 pinn-service/scripts/run_training_experiment.py \
  --output-dir pinn-service/artifacts/checkpoints/pinn_res_split_fourier \
  --architecture res_split \
  --hidden-dim 192 \
  --num-blocks 4 \
  --activation tanh \
  --use-fourier-features \
  --fourier-num-frequencies 6 \
  --fourier-scale 1.0 \
  --epochs 2000 \
  --batch-size 8192 \
  --validation-batch-size 8192 \
  --device cpu
```

## 11. What To Compare Across Experiments

At minimum, compare:

- `supervised_loss`
- `velocity_consistency_loss`
- `wave_residual_loss`
- `thermal_residual_loss`
- `total_loss`
- validation versions of the same metrics when available
- training time
- parameter count

The existing trainer already logs the loss terms per epoch in:

- `metrics.json`
- `metrics.csv`

## 12. Scientific Positioning

The improved `res_split` architecture is a more structured PINN baseline than the original plain MLP, but it is still not a claim of full physical fidelity.

The careful interpretation is:

- the architecture is better aligned with coupled thermoelastic field prediction;
- the residual trunk and split heads are technically safer for this task;
- the current model is still a research prototype and should be evaluated empirically rather than overclaimed.
