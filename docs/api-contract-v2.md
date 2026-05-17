# API Contract v2

**Status:** frozen 2026-05-17. Implementation tracked in
[`api_contract_v2_implementation_plan.md`](api_contract_v2_implementation_plan.md).
Source plan: `api_contract_redesign_plan.pdf` (repo root).

This document is the **single normative reference** for the v2
input–output contract of the AI Directional Prediction prototype.
Anything not described here is out of scope for v2.

> **Scope reminder.** This prototype is a research/MVP system for
> AI-assisted directional prediction and model comparison
> (PINN / FNO / MeshGraphNet / Transformer). It is **not** a
> field-validated thermoelastic simulator. Every v2 response carries
> this disclaimer in `diagnostics.notes[0]`.

---

## 1. High-level shape

A v2 prediction request describes a physical scenario — material,
thermal state, source/probe geometry, observation time, and a
simplified scenario template. The backend resolves the material from
the catalog, derives all geometric and material-derived quantities,
dispatches to the selected model service, and returns a normalised
response in which the physics is grouped by category (thermal,
displacement, directional, temporal) and the diagnostics are kept
separate.

```
client request (minimal v2)
       │
       ▼
backend validation (Pydantic v2 schemas)
       │
       ▼
material resolution + derived_quantities
       │
       ▼
enriched payload → model service (PINN / FNO / MGN / Transformer)
       │
       ▼
remote response → response_normalizer
       │
       ▼
normalised v2 response to client
```

---

## 2. Request

### 2.1 Minimal request (what the frontend sends)

The frontend collects only physically meaningful fields. Material
properties are looked up from the catalog by `medium_id`; the backend
attaches them and derives constants.

```jsonc
{
  "schema_version": "2.0",
  "model": "pinn",
  "medium_id": "sandstone",
  "geometry": {
    "dimension": 2,
    "source": { "x_m": 0.2, "y_m": 0.5 },
    "probe":  { "x_m": 0.8, "y_m": 0.5 }
  },
  "observation": {
    "time_s": 0.1
  },
  "scenario": {
    "thermal_source_type": "point",
    "mechanical_constraint": "free",
    "boundary_condition_type": "prototype_simplified"
  }
}
```

> **What the client sends in v2 (after all locks).** Only four blocks
> are user-facing: `model`, `medium_id`, `geometry` (source + probe
> coordinates inside the 1 m × 1 m domain), and `observation.time_s`.
> Everything else (reference temperature, source temperature,
> frequency, domain size, dimension) is a training-data invariant
> resolved by the backend.

> **Reference temperature is fixed.** The rock initial temperature is
> always 0 °C = **273.15 K** in this prototype (matches the COMSOL
> setup used to generate training data). `reference_temperature_k` is
> therefore **optional** in v2; if omitted the backend uses
> `273.15 K`. Clients that explicitly send a different value get
> rejected with `HTTP 400 / error_code:
> "reference_temperature_override_disabled"`. Frontend does not
> render this field — it is implicit.

> **Domain is fixed at 1 m × 1 m.** All training data was generated on
> a square domain of side 1 m, so the v2 contract locks the spatial
> domain to `lx = ly = 1.0 m`. The request does not carry a
> `domain.size` field — it is implicit and immutable. `geometry.source`
> and `geometry.probe` coordinates must satisfy `0 ≤ x_m ≤ 1.0` and
> `0 ≤ y_m ≤ 1.0`. Out-of-domain coordinates are rejected with
> `HTTP 400 / error_code: "geometry_out_of_domain"`. Explicit
> `domain` overrides in the request are rejected with `HTTP 400 /
> error_code: "domain_override_disabled"`.

> **Source temperature is fixed at 1500 K.** The heating amplitude in
> the training data is constant. `thermal_state.source_temperature_k`
> is therefore implicit too: the backend uses `1500.0 K` always.
> Combined with the fixed reference (273.15 K), the theta is also
> constant at `θ = 1500 − 273.15 = 1226.85 K`. Explicit
> `source_temperature_k` is rejected with `HTTP 400 / error_code:
> "source_temperature_override_disabled"`. **As a consequence, the
> v2 request has no `thermal_state` block at all** — every thermal
> parameter is a training-data invariant.

> **Source frequency is removed entirely.** The v1 contract carried a
> `frequency_hz` field, but the COMSOL simulation is a transient
> thermal-elastic problem with a *step-like* heated rod — there is no
> oscillating source, so `frequency_hz` has no physical referent in
> the training data. (Inside `pinn-service` and `fno-service`,
> `frequency_hz` was used as a heuristic output-magnitude multiplier;
> that code is removed during Phase 4.) v2 has **no source-frequency
> field**. Clients that send `frequency_hz` are rejected with
> `HTTP 400 / error_code: "frequency_field_removed"`.

### 2.2 Required / optional / derived inputs

| Group          | Required (from client)                                           | Optional                            | Derived by backend                                                   |
| -------------- | ---------------------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| Model          | `model`                                                          | `allow_fallback` (default `true`)   | `model_runtime.representation`, routing target URL                   |
| Material       | `medium_id` resolved from catalog                                | manual property override (rare)     | `derived.shear_modulus_pa`, `bulk_modulus_pa`, `lame_lambda_pa`, `C` |
| Thermal state  | *(nothing — fully fixed)*                                        | every thermal field rejected (locked to training data) | `T_ref = 273.15 K`, `T_source = 1500 K`, `θ = 1226.85 K`             |
| Source pulse   | *(nothing — `frequency_hz` removed entirely)*                    | `frequency_hz` rejected with `frequency_field_removed` | none — concept does not exist in v2                                |
| Domain         | *(nothing — fully fixed)*                                        | `domain.size` rejected (locked)                       | `lx = ly = 1.0 m`, `dimension = 2`                                  |
| Geometry       | `geometry.dimension=2`, `source.{x_m,y_m}`, `probe.{x_m,y_m}`    | source direction as comparison vector | `propagation_vector_m`, `distance_m`, `unit_direction`, `azimuth_deg` |
| Observation    | `observation.time_s`                                             | time array for batch prediction     | `time_ms` for v1 compatibility                                       |
| Scenario       | `scenario.{thermal_source_type, mechanical_constraint, boundary_condition_type}` | resolution / boundary labels        | `rect_2d` default domain if omitted                                  |

### 2.3 Enriched payload (backend → model service)

After validation and enrichment the backend dispatches a single
payload shape to every model service. The model service does not
recompute any of the derived quantities — they are authoritative
backend output.

```jsonc
{
  "schema_version": "2.0",
  "model": { "name": "fno", "allow_fallback": true },
  "material": {
    "id": "sandstone",
    "name": "Sandstone",
    "category": "sedimentary siliciclastic",
    "thermoelastic_supported": true,
    "properties": {
      "rho_kg_m3": 2725.0,
      "vp_m_s": 5000.0,
      "vs_m_s": 2850.0,
      "young_modulus_pa": 5.575e10,
      "poisson_ratio": 0.2594,
      "thermal_conductivity_w_mk": 2.25,
      "heat_capacity_j_kgk": 1000.0,
      "thermal_expansion_1_k": 1.16e-5
    },
    "derived_properties": {
      "shear_modulus_pa": 2.213e10,
      "bulk_modulus_pa": 3.861e10,
      "lame_lambda_pa": 2.386e10,
      "volumetric_heat_capacity_j_m3k": 2.725e6
    },
    "metadata": {
      "source_table": "combined_geological_media_parameters.csv",
      "value_type": "mixed",
      "notes": "Midpoint of literature ranges; elastic moduli derived from rho, Vp, Vs under isotropic assumptions."
    }
  },
  "thermal_state": {
    "reference_temperature_k": 273.15,
    "source_temperature_k": 350.0,
    "temperature_perturbation_k": 76.85
  },
  "geometry": {
    "dimension": 2,
    "source": { "x_m": 0.0, "y_m": 0.0 },
    "probe":  { "x_m": 1.0, "y_m": 0.5 },
    "derived": {
      "propagation_vector_m": { "dx": 1.0, "dy": 0.5 },
      "distance_m": 1.118034,
      "unit_direction": { "x": 0.894427, "y": 0.447214 },
      "azimuth_deg": 26.565051
    }
  },
  "observation": { "time_s": 0.1 },
  "model_runtime": {
    "representation": "grid",
    "requested_outputs": ["temperature", "displacement", "direction"]
  }
}
```

`requested_outputs` controls optional payloads (§4.3). `"field_grid"`
must be listed explicitly to receive it — it is FNO-only and off by
default.

---

## 3. Response

### 3.1 Full response shape

```jsonc
{
  "schema_version": "2.0",
  "request_id": "5f3a7c4e-2b8d-4f6a-9c2e-1e7d4a8b9c0f",
  "status": "ok",

  "model": {
    "name": "pinn",
    "version": "pinn-baseline@best_model.pth",
    "route": "/predict",
    "inference_time_ms": 34.2,
    "fallback_used": false,
    "fallback_reason": null
  },

  "material": {
    "id": "sandstone",
    "name": "Sandstone",
    "category": "sedimentary siliciclastic"
  },

  "geometry": {
    "dimension": 2,
    "source": { "x_m": 0.0, "y_m": 0.0 },
    "probe":  { "x_m": 1.0, "y_m": 0.5 },
    "propagation_vector_m": { "dx": 1.0, "dy": 0.5 },
    "unit_direction": { "x": 0.894427, "y": 0.447214 },
    "distance_m": 1.118034,
    "azimuth_deg": 26.565051,
    "azimuth_convention": "atan2(dy, dx), degrees, xy-plane"
  },

  "prediction": {
    "thermal": {
      "temperature_k": {
        "value": 315.2,
        "source": "direct_model_prediction"
      },
      "temperature_perturbation_k": {
        "value": 22.05,
        "reference_temperature_k": 273.15,
        "source": "derived_from_temperature"
      }
    },
    "displacement": {
      "components_m": { "u": 1.2e-06, "v": 3.0e-07 },
      "magnitude_m": 1.237e-06,
      "components_source": "direct_model_prediction",
      "magnitude_source": "derived_from_u_v"
    },
    "directional_response": {
      "distance_m": 1.118034,
      "azimuth_deg": 26.565051,
      "response_magnitude_score": 0.73
    },
    "temporal_response": {
      "travel_time_s": 0.000186,
      "source": "direct_model_prediction"
    }
  },

  "optional_outputs": {
    "confidence_score": null,
    "field_summary": {
      "max_displacement_m": 1.5e-06,
      "max_temperature_perturbation_k": 25.0
    },
    "field_grid": null,
    "strain": null,
    "stress": null
  },

  "diagnostics": {
    "fallback_used": false,
    "fallback_reason": null,
    "warnings": [],
    "notes": [
      "Prototype prediction; not a field-validated thermoelastic simulation."
    ]
  }
}
```

### 3.2 Field ownership and source-of-truth

| Field path                                              | Owner             | Notes                                                                                                |
| ------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------- |
| `schema_version`                                        | backend           | Always `"2.0"`.                                                                                      |
| `request_id`                                            | backend           | UUID per request, surfaced in logs and error responses.                                              |
| `status`                                                | backend           | `"ok"` or `"error"`.                                                                                 |
| `model.*`                                               | backend + service | Runtime metadata. `inference_time_ms` measured around the remote call; `version` from the service.   |
| `material.{id,name,category}`                           | backend           | Slice of the catalog. Full property table not duplicated in the response — frontend can re-query it. |
| `geometry.{source,probe,dimension}`                     | echo of request   | Backend rejects 3D in v2.                                                                            |
| `geometry.{propagation_vector_m,unit_direction,distance_m,azimuth_deg}` | backend (`derived_quantities`) | Single source of truth for all directional math.                                  |
| `prediction.thermal.temperature_k`                      | **model**         | Direct prediction at the probe.                                                                      |
| `prediction.thermal.temperature_perturbation_k`         | backend           | `T − T_ref`. Pulls `T_ref` from the request.                                                         |
| `prediction.displacement.components_m.{u,v}`            | **model**         | Direct prediction at the probe.                                                                      |
| `prediction.displacement.magnitude_m`                   | backend           | `sqrt(u² + v²)`.                                                                                     |
| `prediction.directional_response.distance_m`            | backend           | Duplicate of `geometry.distance_m` — kept here for client convenience.                               |
| `prediction.directional_response.response_magnitude_score` | **model**      | Comparative score in [0, 1] — model-specific, not a physical quantity.                               |
| `prediction.temporal_response.travel_time_s`            | **model** (mandatory) | All four routes must populate it. MGN fallback uses the analytical `r/V_p`.                     |
| `optional_outputs.field_summary.*`                      | model or backend  | Backward compatibility with v1 consumers.                                                            |
| `optional_outputs.field_grid`                           | **FNO only**, opt-in | `null` unless the request lists `"field_grid"` in `model_runtime.requested_outputs`.             |
| `optional_outputs.{confidence_score, strain, stress}`   | declared          | Always `null` in v2. Implementations deferred to v2.1.                                               |
| `diagnostics.*`                                         | backend           | `notes[0]` carries the canonical prototype disclaimer.                                               |

---

## 4. Units, conventions, and limits

### 4.1 Units

| Quantity                                                | Unit               |
| ------------------------------------------------------- | ------------------ |
| Temperature, temperature perturbation                   | kelvin (K)         |
| Position, distance                                      | metre (m)          |
| Displacement                                            | metre (m)          |
| Time                                                    | second (s)         |
| Inference latency                                       | millisecond (ms)   |
| Density `rho_kg_m3`                                     | kg / m³            |
| Wave speeds `vp_m_s`, `vs_m_s`                          | m / s              |
| Elastic moduli                                          | pascal (Pa)        |
| Thermal conductivity                                    | W / (m · K)        |
| Heat capacity                                           | J / (kg · K)       |
| Thermal expansion                                       | 1 / K              |
| Azimuth                                                 | degrees, xy-plane  |

### 4.2 Geometry conventions

- Coordinate system: 2D Cartesian, `+x` to the right, `+y` upward.
- Azimuth: `atan2(dy, dx)` where `(dx, dy) = probe − source`,
  reported in degrees in `(-180, 180]`.
- `rect_2d` is the only `geometry.dimension` value accepted in v2.
  3D is reserved for future extension; the backend rejects
  `dimension=3` with HTTP 400.
- The request is rejected with HTTP 400 if `source == probe` or
  if either coordinate is non-finite.

### 4.3 `requested_outputs` knob

| Token             | Effect                                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `"temperature"`   | Default. Always populates `prediction.thermal`.                                                                            |
| `"displacement"`  | Default. Always populates `prediction.displacement`.                                                                       |
| `"direction"`     | Default. Always populates `prediction.directional_response` and `geometry.*` derived fields.                               |
| `"field_grid"`    | Opt-in. Only honoured by FNO. Capped at `Nx × Ny ≤ 128 × 128` (~1.5 MB response). All other services leave `field_grid: null`. |

### 4.4 Material catalog

- Source of truth: `chapters/tables/combined_geological_media_parameters.csv`
  (10 materials). The backend catalog
  `backend/data/media/catalog.json` is a JSON projection of that
  table.
- Materials without a value for `alpha_1_K` (thermal expansion) are
  flagged `thermoelastic_supported: false`. The backend rejects
  thermoelastic requests against them with HTTP 400 and a
  `error_code: "material_thermoelastic_unsupported"`. The frontend
  greys out the submit button for these materials.
- Elastic moduli (`young_modulus_pa`, `poisson_ratio`, `mu_Pa`,
  `K_Pa`, `lambda_Pa`) come straight from the CSV — they were
  derived from midpoint `rho`, `Vp`, `Vs` under isotropic
  assumptions during table preparation. The backend verifies on
  startup that `|E_csv − 9 K μ / (3 K + μ)| / E_csv < 0.01` and
  logs a warning on mismatch.

---

## 5. Backward compatibility

- `POST /api/v1/predictions` is the only endpoint URL. It accepts
  both v1 and v2 payloads. The backend dispatches based on
  `schema_version`:
  - missing or `"1.0"` → v1 path, v1 response (unchanged from today).
  - `"2.0"` → v2 path, v2 response.
- A query flag `?contract=v2` can be added to a v1-shaped request to
  force a v2-shaped response (the backend re-normalises). This is
  the channel the frontend uses while the form is still v1 but the
  renderer is v2.
- `ResponseNormalizer` accepts both remote v1 (flat) and remote v2
  payloads from the model services and produces a v2 client
  response in either case. Old flat fields land in
  `optional_outputs.field_summary.*` and
  `prediction.directional_response.*`.

---

## 6. Example `curl`

```bash
curl -s -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  --data '{
    "schema_version": "2.0",
    "model": "pinn",
    "medium_id": "sandstone",
    "geometry": {
      "dimension": 2,
      "source": { "x_m": 0.2, "y_m": 0.5 },
      "probe":  { "x_m": 0.8, "y_m": 0.5 }
    },
    "observation": { "time_s": 0.1 },
    "scenario": {
      "thermal_source_type": "point",
      "mechanical_constraint": "free",
      "boundary_condition_type": "prototype_simplified"
    }
  }'
```

A successful response is the JSON document of §3.1.

---

## 7. Errors

All errors return HTTP 4xx/5xx and a JSON body of the form:

```jsonc
{
  "schema_version": "2.0",
  "status": "error",
  "error_code": "material_thermoelastic_unsupported",
  "error_message": "Material 'basalt' has no thermal_expansion_1_k value; thermoelastic predictions are not supported for this medium.",
  "request_id": "5f3a7c4e-2b8d-4f6a-9c2e-1e7d4a8b9c0f"
}
```

| `error_code`                              | HTTP | Cause                                                                |
| ----------------------------------------- | ---- | -------------------------------------------------------------------- |
| `schema_validation_failed`                | 422  | Pydantic rejected the payload.                                       |
| `unknown_medium`                          | 404  | `medium_id` not in the catalog.                                      |
| `material_thermoelastic_unsupported`      | 400  | Catalog entry has `thermoelastic_supported: false`.                  |
| `reference_temperature_override_disabled` | 400  | Client sent `reference_temperature_k`; v2 fixes it at 273.15 K.      |
| `source_temperature_override_disabled`    | 400  | Client sent `source_temperature_k`; v2 fixes it at 1500 K.           |
| `frequency_field_removed`                 | 400  | Client sent `frequency_hz`; the field is removed entirely in v2 (no physical referent in COMSOL training data). |
| `domain_override_disabled`                | 400  | Client sent `domain.size` or `domain.resolution`; v2 fixes domain at 1 m × 1 m. |
| `geometry_out_of_domain`                  | 400  | `source` or `probe` coordinate outside `[0, 1] m`.                   |
| `invalid_geometry`                        | 400  | `source == probe`, non-finite coords, or `dimension != 2`.           |
| `model_route_unavailable`                 | 503  | Selected model service is down and `allow_fallback: false`.          |
| `model_route_error`                       | 502  | Model service returned non-2xx; details surfaced in `diagnostics`.   |
| `internal_error`                          | 500  | Unhandled.                                                           |

---

## 8. Thesis-safe wording (PDF §13, normative)

The practical software prototype uses a normalized two-dimensional
input–output contract for AI-assisted directional prediction of
thermoelastic wave behavior. The input side describes the geological
material, thermal state, source–probe geometry, and observation time.
The backend derives the propagation vector, distance, unit direction,
azimuth, temperature perturbation, and selected material constants in
order to keep these definitions consistent across model routes. The
output side reports predicted temperature response, temperature
perturbation, in-plane displacement components, displacement
magnitude, directional geometry, predicted travel time, and
model-comparison metadata. Stress and strain are not mandatory direct
outputs of every route; they may be included only as derived
diagnostics when sufficient spatial information is available. This
contract supports comparison between predictive model components
without presenting the prototype as a full field-validated
thermoelastic simulator.

This paragraph is the canonical thesis-facing description of the
contract. It is also surfaced verbatim in `diagnostics.notes` (first
sentence only) so any consumer of the API sees the scope language.

---

## 9. Changelog

| Date       | Change                                                                                              |
| ---------- | --------------------------------------------------------------------------------------------------- |
| 2026-05-17 | v2.0 frozen. Three open questions resolved (catalog from CSV, hybrid optional outputs, ?contract=v2 flag). |
