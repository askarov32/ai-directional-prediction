# Prediction Contract And PINN Training Formulas

This document collects the public prediction API contract and the current coupled thermoelastic PINN training objective.

## Prediction Request

Endpoint:

```text
POST /api/v1/predictions
```

Request body:

```json
{
  "model": "pinn",
  "medium_id": "granite",
  "scenario": {
    "temperature_c": 120.0,
    "pressure_mpa": 35.0,
    "time_ms": 12.0
  },
  "source": {
    "type": "thermal_pulse",
    "x": 0.15,
    "y": 0.4,
    "z": 0.0,
    "amplitude": 1.0,
    "frequency_hz": 50.0,
    "direction": [1.0, 0.0, 0.0]
  },
  "probe": {
    "x": 0.7,
    "y": 0.55,
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
    },
    "boundary_conditions": {
      "left": "fixed",
      "right": "free",
      "top": "insulated",
      "bottom": "insulated"
    }
  }
}
```

Supported `model` values:

- `meshgraphnet`
- `fno`
- `pinn`

The backend resolves `medium_id`, merges the selected rock preset into the request, validates scenario ranges, routes the payload to the selected model service, and normalizes the model response.

## Curl Example

```bash
curl -s -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  --data '{
    "model": "pinn",
    "medium_id": "granite",
    "scenario": {
      "temperature_c": 120.0,
      "pressure_mpa": 35.0,
      "time_ms": 12.0
    },
    "source": {
      "type": "thermal_pulse",
      "x": 0.15,
      "y": 0.4,
      "z": 0.0,
      "amplitude": 1.0,
      "frequency_hz": 50.0,
      "direction": [1.0, 0.0, 0.0]
    },
    "probe": {
      "x": 0.7,
      "y": 0.55,
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
      },
      "boundary_conditions": {
        "left": "fixed",
        "right": "free",
        "top": "insulated",
        "bottom": "insulated"
      }
    }
  }'
```

## Normalized Response

```json
{
  "model": "pinn",
  "medium": {
    "id": "granite",
    "name": "Granite",
    "category": "igneous"
  },
  "prediction": {
    "direction_vector": [0.821, 0.571, 0.0],
    "azimuth_deg": 34.8,
    "elevation_deg": 0.0,
    "magnitude": 0.914,
    "wave_type": "physics_informed",
    "travel_time_ms": 0.111
  },
  "field_summary": {
    "max_displacement": 0.001327,
    "max_temperature_perturbation": 1.742
  },
  "meta": {
    "model_version": "pinn-baseline@best_model.pth",
    "latency_ms": 42,
    "request_id": "f4c06f4f-8b5b-48c1-b0f9-49b4c1d7b5f0"
  }
}
```

## PINN Training Inputs And Outputs

Network input:

```latex
X = [x, y, z, t, E, \nu, \rho, \alpha, k, C_p]
```

Network output:

```latex
\hat{Y} = [\hat{T}, \hat{u}, \hat{v}, \hat{w}]
```

The current implementation treats material parameters as locally homogeneous pointwise features. It does not take spatial derivatives of `E`, `nu`, `rho`, `alpha`, `k`, or `Cp`.

## Coupled Thermoelastic Training Objective

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

Small strain tensor:

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

Elastic wave residual:

```latex
R_i^{wave}
=
\rho
\frac{\partial^2 u_i}{\partial t^2}
-
\frac{\partial \sigma_{ij}}{\partial x_j}
```

```latex
\mathcal{L}_{wave}
=
\operatorname{mean}(R_u^2 + R_v^2 + R_w^2)
```

Coupled thermal residual:

```latex
R_T =
\rho C_p
\frac{\partial T}{\partial t}
-
k\nabla^2T
+
\gamma T_0
\frac{\partial \varepsilon_{kk}}{\partial t}
```

```latex
\mathcal{L}_{temp}
=
\operatorname{mean}(R_T^2)
```

Total loss:

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

```latex
\lambda_{sup}=1.0,\qquad
\lambda_{vel}=0.25,\qquad
\lambda_{wave}=0.1,\qquad
\lambda_{temp}=0.05
```
