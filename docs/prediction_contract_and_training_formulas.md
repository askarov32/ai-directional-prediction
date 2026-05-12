# Prediction Contract And PINN Training Formulas

This document collects two things in one place:

- the exact unified prediction request contract used by the frontend and backend;
- the current LaTeX formulas for the baseline PINN training objective.

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
"sandstone_medium"
"granite"
"basalt"
```

### `scenario`

Fields:

- `temperature_c`: `float`, range `[-273.15, 2000]`
- `pressure_mpa`: `float`, range `(0, 5000]`
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

The total loss is:

```latex
\mathcal{L}_{total}
=
\lambda_{sup}\mathcal{L}_{sup}
+
\lambda_{vel}\mathcal{L}_{vel}
+
\lambda_{temp}\mathcal{L}_{temp}
```

Current default weights:

```latex
\lambda_{sup}=1.0,\qquad
\lambda_{vel}=0.25,\qquad
\lambda_{temp}=0.05
```

Supervised field loss:

```latex
\mathcal{L}_{sup}
=
\operatorname{MSE}
\left(
[\hat{T},\hat{u},\hat{v},\hat{w}],
[T,u,v,w]
\right)
```

Velocity-consistency loss:

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

Thermal diffusivity:

```latex
a = \frac{k}{\rho C_p}
```

Thermal residual:

```latex
R_T =
\frac{\partial \hat{T}}{\partial t}
-
a
\left(
\frac{\partial^2 \hat{T}}{\partial x^2}
+
\frac{\partial^2 \hat{T}}{\partial y^2}
+
\frac{\partial^2 \hat{T}}{\partial z^2}
\right)
```

Thermal residual loss:

```latex
\mathcal{L}_{temp}
=
\operatorname{mean}(R_T^2)
```

## 7. Compact One-Line Formula

```latex
\mathcal{L}_{total}
=
1.0 \cdot \operatorname{MSE}([\hat{T},\hat{u},\hat{v},\hat{w}],[T,u,v,w])
+
0.25 \cdot \operatorname{MSE}
\left(
\left[
\frac{\partial \hat{u}}{\partial t},
\frac{\partial \hat{v}}{\partial t},
\frac{\partial \hat{w}}{\partial t}
\right],
[u_t,v_t,w_t]
\right)
+
0.05 \cdot \operatorname{mean}(R_T^2)
```

## 8. Important Scientific Note

These formulas describe the current baseline in this repository.

This is not yet a full coupled thermoelastic PINN with:

- elastic momentum residual;
- stress-strain constitutive residual;
- thermoelastic coupling term;
- boundary-condition loss;
- initial-condition loss;
- collocation-point residual sampling.

Those are the next scientific/modeling steps.
