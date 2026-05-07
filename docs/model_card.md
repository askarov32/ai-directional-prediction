# Model Card

## Project

Thermoelastic Direction Predictor MVP for:

> AI directional prediction of the propagation of thermoelastic waves in geological media.

## Intended Use

This project is intended for:

- thesis/pre-defense demonstration;
- local experimentation with API orchestration around model services;
- comparing model routes behind one unified prediction contract;
- demonstrating how geological medium presets and scenario inputs are merged before inference.

## Not Intended Use

Do not use this MVP for:

- engineering safety decisions;
- field-scale geophysical interpretation;
- validated seismic or thermoelastic forecasting;
- production scientific claims without additional validation;
- replacing calibrated numerical solvers or laboratory measurements.

## Model Routes

### MeshGraphNet

Current local implementation:

- deterministic mock FastAPI service;
- accepts graph-oriented payload shape;
- returns synthetic, frontend-friendly values.

Expected future implementation:

- graph/mesh neural operator over nodes, edges, material fields, source/probe geometry.

### FNO

Current local implementation:

- deterministic mock FastAPI service;
- accepts grid-oriented payload shape;
- returns synthetic, frontend-friendly values.

Expected future implementation:

- Fourier Neural Operator over regular fields/grids with material and source channels.

### PINN

Current local implementation:

- real PyTorch checkpoint loaded by `pinn-service`;
- startup smoke inference required for readiness;
- neural outputs are `temperature_k`, `disp_x`, `disp_y`, `disp_z`;
- final direction is hybrid neural output plus geometry/material postprocessing.

## Inputs

Public backend request inputs:

- selected model route;
- geological medium id;
- temperature, pressure, time;
- source type, coordinates, amplitude, frequency, direction;
- probe coordinates;
- domain type, size, resolution, boundary conditions.

Medium properties loaded from catalog:

- density `rho`;
- total/effective porosity;
- P-wave velocity `vp`;
- S-wave velocity `vs`;
- thermal conductivity;
- heat capacity;
- thermal expansion.

## Outputs

Normalized backend response:

- direction vector;
- azimuth;
- elevation;
- magnitude;
- wave type;
- travel time;
- max displacement;
- max temperature perturbation;
- model version;
- latency;
- request id.

PINN service also exposes raw/diagnostic fields:

- `model_outputs`;
- `postprocessed_prediction`;
- `diagnostics`.

## Training Data

The PINN baseline is prepared from COMSOL CSV exports:

- `data_materials.csv`;
- `data_temperature.csv`;
- `data_displacement.csv`;
- `data_stress_1.csv`;
- `data_stress_2.csv`;
- `data_stress_3.csv`.

The data pipeline builds:

- structured field dataset;
- flattened training samples;
- scaler metadata;
- checkpoint metrics.

## Evaluation Data And Metrics

Current MVP status:

- no independent validation dataset is yet documented as final scientific evidence;
- checkpoint training stores loss metrics;
- readiness verifies checkpoint load, feature alignment, output shape, and finite inference outputs;
- frontend/backend smoke tests verify API behavior, not scientific validity.

Recommended next metrics:

- held-out trajectory/field error;
- direction angular error in degrees;
- travel-time error;
- displacement/temperature perturbation MAE/RMSE;
- robustness across media, pressure, temperature, and source parameters.

## Known Limitations

- MeshGraphNet and FNO are mocks in the default local stack.
- PINN is a first baseline, not a complete coupled thermoelastic PDE solver.
- Medium presets are starter values and should be replaced with validated references.
- Current MVP is 2D-first in the UI, though the request shape supports 3D.
- PINN direction combines neural outputs with deterministic postprocessing.
- Scientific generalization outside the prepared data distribution is not established.

## Recommended Demo Statement

Use this wording:

> The MVP demonstrates a clean full-stack architecture for thermoelastic-wave direction prediction with model routing to MeshGraphNet, FNO, and PINN services. MeshGraphNet and FNO are currently deterministic mock services, while PINN uses a real checkpoint-based baseline with readiness diagnostics. The current prediction output is suitable for architecture and thesis-demo illustration, not final scientific validation.
