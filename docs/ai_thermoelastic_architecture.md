# AI Thermoelastic Prediction Architecture

## 1. What follows directly from the presentation

From the slides, the core physically meaningful inputs are:

- geological medium class
- density `rho`
- porosity
- `Vp`
- `Vs`
- thermal conductivity `k`
- heat capacity `cp`
- thermal expansion coefficient `alpha`
- temperature
- object coordinates in the domain

The target outputs are:

- predicted propagation direction
- approximate thermoelastic-wave response summary

## 2. Recommended product scope

For a usable MVP, it is better to separate inputs into three levels.

### Level A: required for the first demo

These are the minimum inputs the frontend should collect:

- `rock_type_id`
- `temperature_c`
- `point.x`
- `point.y`
- `point.z`
- `model_type` (`meshgraphnet`, `fno`, `pinn`)

These are the minimum physical values the backend must resolve before model inference:

- `rho`
- `porosity_total`
- `porosity_effective`
- `vp`
- `vs`
- `thermal_conductivity`
- `heat_capacity`
- `thermal_expansion`

### Level B: strongly recommended for realistic prediction

Without these, “directional propagation” will be too underspecified:

- `pressure_mpa`
- `source_type` (`point`, `line`, `plane`)
- `source_point`
- `source_direction`
- `wave_mode` (`p`, `s`, `coupled`)
- `reference_temperature_c`

### Level C: useful for research mode

- `domain_size`
- `time_horizon_ms`
- `time_step_ms`
- `boundary_condition`
- `anisotropy_tensor`
- `fluid_saturation`
- `sample_id`

## 3. Frontend data model

For the native frontend, the cleanest approach is:

- mobile UI in `SwiftUI` or `Jetpack Compose`
- shared API contract generated from OpenAPI
- local catalog cache for rock presets

The screen should expose:

1. Rock type picker
2. Temperature input
3. Pressure input
4. Coordinate inputs `x/y/z`
5. Source configuration
6. Model selector
7. Predict button
8. Result card with direction, confidence, and response values

### Frontend payload to FastAPI

```json
{
  "model_type": "meshgraphnet",
  "scenario_name": "sandstone_test_01",
  "rock_type_id": "sandstone_medium",
  "temperature_c": 120.0,
  "pressure_mpa": 35.0,
  "reference_temperature_c": 20.0,
  "point": { "x": 12.5, "y": 8.0, "z": 3.0 },
  "source": {
    "source_type": "point",
    "source_point": { "x": 10.0, "y": 8.0, "z": 3.0 },
    "source_direction": { "x": 1.0, "y": 0.0, "z": 0.0 },
    "wave_mode": "coupled"
  }
}
```

## 4. How to store geological media

Recommended approach for the first version:

- store preset rock classes in `config/rock_catalog.json`
- each record contains a stable `id`, display name, default physical parameters, value ranges, and metadata source
- backend loads this catalog on startup
- frontend reads available rocks from `GET /catalog/rocks`

Why JSON first:

- easy to version in Git
- easy to seed into Docker
- easy for non-backend teammates to edit
- later can be moved into Postgres without changing the public API

### Suggested rock record shape

```json
{
  "id": "sandstone_medium",
  "name": "Sandstone (medium)",
  "category": "sedimentary",
  "properties": {
    "rho": 2684.0,
    "porosity_total": 0.34,
    "porosity_effective": 0.27,
    "vp": 6.17,
    "vs": 3.20,
    "thermal_conductivity": 2.5,
    "heat_capacity": 850.0,
    "thermal_expansion": 0.000012
  },
  "ranges": {
    "temperature_c": [-20, 300],
    "pressure_mpa": [0.1, 1500]
  },
  "metadata": {
    "source": "presentation_seed",
    "notes": "Starter preset derived from slides and should be refined with lab/reference data."
  }
}
```

## 5. Backend contract design

The backend should not let each model define its own public API. Instead:

- frontend sends one canonical request to FastAPI
- FastAPI enriches the request with rock properties from the catalog
- FastAPI transforms the canonical request into model-specific payloads
- each model service stays behind an adapter layer

### Canonical backend request

```json
{
  "model_type": "fno",
  "rock": {
    "id": "sandstone_medium",
    "rho": 2684.0,
    "porosity_total": 0.34,
    "porosity_effective": 0.27,
    "vp": 6.17,
    "vs": 3.20,
    "thermal_conductivity": 2.5,
    "heat_capacity": 850.0,
    "thermal_expansion": 0.000012
  },
  "scenario": {
    "temperature_c": 120.0,
    "reference_temperature_c": 20.0,
    "pressure_mpa": 35.0,
    "point": { "x": 12.5, "y": 8.0, "z": 3.0 },
    "source": {
      "source_type": "point",
      "source_point": { "x": 10.0, "y": 8.0, "z": 3.0 },
      "source_direction": { "x": 1.0, "y": 0.0, "z": 0.0 },
      "wave_mode": "coupled"
    }
  }
}
```

### Unified response from FastAPI

```json
{
  "model_type": "fno",
  "prediction": {
    "direction_vector": { "x": 0.92, "y": 0.31, "z": 0.21 },
    "direction_azimuth_deg": 18.6,
    "direction_elevation_deg": 12.1,
    "response_amplitude": 0.71,
    "response_time_ms": 4.8,
    "confidence": 0.84
  },
  "meta": {
    "rock_type_id": "sandstone_medium",
    "served_by": "fno-service",
    "model_version": "v1",
    "request_id": "uuid"
  }
}
```

## 6. What each model actually needs

The public API should be unified, but internal model inputs should differ.

### MeshGraphNet

Best when the medium is represented as a mesh or graph.

Internal input:

- node coordinates
- node features:
  - temperature
  - pressure
  - `rho`
  - porosity
  - `Vp`
  - `Vs`
  - `k`
  - `cp`
  - `alpha`
- edge index
- edge features:
  - distance
  - relative orientation
  - material contrast
- source node or nearest source location

Use when:

- geometry is irregular
- you care about local heterogeneity
- you already have mesh-based simulation data

### FNO

Best when the problem is placed on a regular grid.

Internal input:

- tensor grid `X x Y x Z`
- per-cell channels:
  - temperature
  - pressure
  - `rho`
  - porosity
  - `Vp`
  - `Vs`
  - `k`
  - `cp`
  - `alpha`
- optional source mask channel
- optional coordinate channels

Use when:

- the domain can be rasterized
- you want fast operator-style inference
- training data is generated on regular grids

### PINN

Best when you want direct physics constraints in training/inference.

Internal input:

- continuous coordinates `(x, y, z, t)`
- rock/material parameters:
  - `rho`
  - `lambda`
  - `mu`
  - `k`
  - `cp`
  - `alpha`
- initial and boundary conditions
- source terms

Use when:

- physical consistency matters more than raw throughput
- data is limited
- you need interpolation at arbitrary coordinates

## 7. Recommended backend adapters

Use this internal adapter split:

- `catalog_service`: returns available rock types and defaults
- `scenario_service`: validates and enriches requests
- `model_router`: picks target model host
- `meshgraphnet_adapter`
- `fno_adapter`
- `pinn_adapter`

Each adapter should accept the same canonical Python object and map it to the target model payload.

## 8. FastAPI endpoints

Recommended API surface:

- `GET /health`
- `GET /catalog/rocks`
- `GET /catalog/rocks/{rock_id}`
- `POST /predict`
- `POST /predict/batch`
- `GET /models`

### `GET /models`

Should return which model services are configured and reachable:

```json
{
  "models": [
    { "type": "meshgraphnet", "base_url": "http://meshgraphnet:8001", "healthy": true },
    { "type": "fno", "base_url": "http://fno:8002", "healthy": true },
    { "type": "pinn", "base_url": "http://pinn:8003", "healthy": true }
  ]
}
```

## 9. Docker architecture

Recommended local setup:

- `api` container for FastAPI gateway
- `meshgraphnet` container
- `fno` container
- `pinn` container
- optional `postgres` later

For the first phase, the model host URLs should come from environment variables:

- `MESHGRAPHNET_URL`
- `FNO_URL`
- `PINN_URL`

This lets teammates replace model services without changing application code.

## 10. Final recommendation for the MVP

If you want the fastest path to a strong demo:

1. Keep rock presets in JSON
2. Build one canonical `POST /predict`
3. Let FastAPI enrich the scenario with preset physics
4. Start with one common response format
5. Hide model-specific differences behind adapters
6. Treat `MeshGraphNet` as the first real model, and keep `FNO`/`PINN` behind mock or placeholder services until trained models are ready

## 11. Important scientific note

Based on the presentation alone, `temperature + coordinates` is not enough for a physically credible directional prediction. At minimum, add:

- pressure
- source location
- source direction or excitation type
- wave mode

Otherwise, the model will be forced to invent missing physics from too little context.
