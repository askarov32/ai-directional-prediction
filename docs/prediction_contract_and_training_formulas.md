# Prediction Contract And PINN Training Formulas

This document collects two things in one place:

- the exact unified prediction request contract used by the frontend and backend;
- the current coupled thermoelastic PINN training objective.

## 1. Unified Prediction Request Contract

Endpoint:

```text
POST /api/v1/predictions
```

Top-level fields:

- `model`
- `medium_id`
- `scenario`
- `source`
- `probe`
- `domain`

### `model`

Allowed values:

```json
"meshgraphnet" | "fno" | "pinn"
```

### `medium_id`

String id of the geological medium preset loaded from:

```text
backend/data/media/catalog.json
```

Examples:

```json
"granite"
"limestone"
"sandstone_medium"
"basalt"
```

### `scenario`

Fields:

- `temperature_c`: `float`, range is validated against the selected medium preset
- `pressure_mpa`: `float`, range is validated against the selected medium preset
- `time_ms`: `float`, range `(0, 60000]`

### `source`

Fields:

- `type`: `string`
- `x`: `float >= 0`
- `y`: `float >= 0`
- `z`: `float >= 0`
- `amplitude`: `float`, range `(0, 1000000]`
- `frequency_hz`: `float`, range `(0, 1000000]`
- `direction`: array of exactly 3 numeric values

Important:

- the backend validates that the direction vector has exactly 3 components;
- the backend normalizes the direction vector before sending it further;
- source coordinates must lie inside the configured domain bounds.

### `probe`

Fields:

- `x`: `float >= 0`
- `y`: `float >= 0`
- `z`: `float >= 0`

Important:

- probe coordinates must also lie inside the configured domain bounds.

### `domain`

Fields:

- `type`: `"rect_2d"` or `"rect_3d"`
- `size`
- `resolution`
- `boundary_conditions`

#### `domain.size`

- `lx`: `float`, range `(0, 10000]`
- `ly`: `float`, range `(0, 10000]`
- `lz`: `float`, range `[0, 10000]`

#### `domain.resolution`

- `nx`: `int`, range `[2, 2048]`
- `ny`: `int`, range `[2, 2048]`
- `nz`: `int`, range `[1, 512]`

#### `domain.boundary_conditions`

Fields:

- `left`: `string`
- `right`: `string`
- `top`: `string`
- `bottom`: `string`
- `front`: `string | null`
- `back`: `string | null`

Additional domain rules:

- if `domain.type == "rect_2d"`, then `lz = 0` and `nz = 1`;
- if `domain.type == "rect_3d"`, then `lz > 0`.

## 2. Example Request JSON

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

The backend resolves `medium_id`, merges the selected rock preset into the request, validates scenario ranges, routes the payload to the selected model service, and normalizes the model response.

## 3. Example `curl` Request

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

## 4. Example Normalized Response JSON

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

## 5. What The Backend Sends To Model Services

The public request above is not the final internal payload.

Before calling the selected model service, the backend:

1. validates the request;
2. resolves `medium_id`;
3. loads preset rock properties from the media catalog;
4. validates temperature and pressure against the medium range;
5. builds an enriched payload for the selected model route.

The enriched payload contains:

- medium summary;
- medium physical properties;
- medium allowed ranges;
- medium metadata;
- scenario;
- source;
- probe;
- domain;
- route-specific `representation`.

For example, PINN receives:

```json
{
  "medium": {
    "id": "granite",
    "name": "Granite",
    "category": "igneous",
    "properties": {
      "rho": 2650.0,
      "porosity_total": 0.015,
      "porosity_effective": 0.005,
      "vp": 5.95,
      "vs": 3.4,
      "thermal_conductivity": 2.7,
      "heat_capacity": 790.0,
      "thermal_expansion": 0.0000075
    },
    "ranges": {
      "temperature_c": [-20.0, 500.0],
      "pressure_mpa": [0.1, 2000.0]
    },
    "metadata": {
      "source": "mvp_catalog_extension",
      "notes": "Starter preset for hard crystalline rock. Replace with validated laboratory values when available."
    }
  },
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
  },
  "representation": "physics_informed",
  "routing_hint": "pinn"
}
```

## 6. Current PINN Training Formulas In LaTeX

Current training input:

```latex
X = [x, y, z, t, E, \nu, \rho, \alpha, k, C_p]
```

Current primary network output:

```latex
\hat{Y} = [\hat{T}, \hat{u}, \hat{v}, \hat{w}]
```

The current implementation treats material parameters as locally homogeneous pointwise features. It does not take spatial derivatives of `E`, `nu`, `rho`, `alpha`, `k`, or `Cp`.

### Lame parameters

```latex
\mu = \frac{E}{2(1+\nu)}
```

```latex
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
```

### Thermoelastic coupling

```latex
\gamma = (3\lambda + 2\mu)\alpha
```

### Small strain tensor

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

```latex
\varepsilon_{kk}
=
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
+
\frac{\partial w}{\partial z}
```

### Thermoelastic stress

```latex
\sigma_{ij}
=
\lambda \delta_{ij}\varepsilon_{kk}
+
2\mu\varepsilon_{ij}
-
\gamma \delta_{ij}(T - T_0)
```

### Elastic wave residual

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

### Coupled thermal residual

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

### Supervised field loss

```latex
\mathcal{L}_{sup}
=
\operatorname{MSE}
\left(
[\hat{T},\hat{u},\hat{v},\hat{w}],
[T,u,v,w]
\right)
```

### Velocity-consistency loss

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
[u_t,v_t,w_t]
\right)
```

### Total loss

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

Current default weights:

```latex
\lambda_{sup}=1.0,\qquad
\lambda_{vel}=0.25,\qquad
\lambda_{wave}=0.1,\qquad
\lambda_{temp}=0.05
```

## 7. Scale Correction For Autograd Derivatives

Inputs and outputs are standardized for neural network training, but PDE residuals are computed in physical units.

For a standardized coordinate:

```latex
x_s = \frac{x - \mu_x}{\sigma_x}
```

the physical derivative is corrected as:

```latex
\frac{\partial}{\partial x}
=
\frac{1}{\sigma_x}
\frac{\partial}{\partial x_s}
```

and:

```latex
\frac{\partial^2}{\partial x^2}
=
\frac{1}{\sigma_x^2}
\frac{\partial^2}{\partial x_s^2}
```

The same correction is applied for `y`, `z`, and `t`.

## 8. Important Scientific Note

These formulas describe the current coupled thermoelastic training baseline in this repository.

The current implementation includes:

- supervised field fitting for temperature and displacement;
- velocity-consistency loss;
- elastic wave residual;
- coupled thermal residual;
- pointwise locally homogeneous material parameters.

The next scientific/modeling steps are:

- explicit initial-condition loss;
- explicit boundary-condition loss;
- separate collocation-point residual sampling;
- stronger nondimensionalization or adaptive loss weighting for residual balance.
