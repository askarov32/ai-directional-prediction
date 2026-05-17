# API Contract v2 — Implementation Checklist

**Source plan:** `api_contract_redesign_plan.pdf` (root of repo, commit `446950f`).
**Status (2026-05-17):** documented, not yet implemented. No code changes
yet — this file freezes the order of work first.

**Scope reminder.** This is a research/MVP prototype for AI-assisted
directional prediction and model comparison (PINN / FNO / MeshGraphNet /
Transformer). It is *not* a field-validated thermoelastic simulator. The
contract must preserve that wording in `diagnostics.notes`.

**Migration principle (PDF §14).** Do not start by rewriting the
frontend. The order is: freeze `docs/api-contract-v2.md` → backend
schemas → normalizer maps old + new → model services → frontend → tests.
Old services must keep running until the normalizer can produce v2 from
their v1 responses.

---

## Phase 0 — Contract document (no code)

- [ ] `docs/api-contract-v2.md` — full request and response examples,
  units, field ownership table (model / backend / derived). Lift from
  PDF §4.1, §4.2, §5, §5.1.
- [ ] Add curl example (PDF §11) to the same doc.
- [ ] Add "thesis-safe wording" block (PDF §13) so reviewers see scope
  language from day one.
- [ ] Keep this file (`api_contract_v2_implementation_plan.md`) and
  `docs/api-contract-v2.md` cross-linked.

**Done when:** another contributor can implement a v2 client from
`docs/api-contract-v2.md` alone, without reading the PDF.

---

## Phase 1 — Backend schemas (additive, v1 stays)

Touch only `backend/app/`:

- [ ] `backend/app/schemas/prediction.py`
  - Add `PredictionRequestV2Schema` (top-level fields: `schema_version`,
    `model`, `medium_id`, `thermal_state`, `geometry`, `observation`,
    `scenario`).
  - Add sub-schemas: `ThermalStateSchema` (reference_temperature_k,
    source_temperature_k, optional temperature_perturbation_k),
    `Geometry2DSchema` (dimension, source{x_m,y_m},
    probe{x_m,y_m}), `ObservationSchema` (time_s),
    `ScenarioPrototypeSchema` (thermal_source_type,
    mechanical_constraint, boundary_condition_type).
  - Add `PredictionResponseV2Schema` with the four nested blocks of
    PDF §5: `prediction.thermal`, `prediction.displacement`,
    `prediction.directional_response`, `optional_outputs`,
    `diagnostics`, plus top-level `model`/`material`/`geometry`.
  - Keep `PredictionRequestSchema` / `PredictionResponseSchema` (v1)
    untouched.
- [ ] `backend/app/domain/entities/prediction.py`
  - Add dataclasses `ThermalState`, `Geometry2D`, `Observation`,
    `DerivedGeometry`, `NormalizedPredictionOutput`. These mirror the
    schemas but are free of FastAPI/Pydantic noise so the use case can
    work in pure Python.
- [ ] `backend/app/domain/services/derived_quantities.py` **(new file)**
  - `compute_theta(thermal_state)`,
    `compute_distance(source, probe)`,
    `compute_unit_direction(...)`, `compute_azimuth_deg(...)`,
    `compute_displacement_magnitude(u, v)`,
    `derive_elastic_constants(material)` (PDF §6.2 — implements both
    E/ν and Vp/Vs/ρ paths).
  - Unit-tested in isolation (no FastAPI, no IO).
- [ ] `backend/data/media/catalog.json`
  - Add `young_modulus_pa`, `poisson_ratio` where derivable, OR add a
    `derivation` field explaining the Vp/Vs/ρ formula source. Do not
    silently change any existing numeric value.

**Done when:** `pytest backend/tests/test_prediction_schema.py` (new
v2 cases) passes and the v1 suite is still green.

---

## Phase 2 — Normalizer + use case

- [ ] `backend/app/infrastructure/adapters/remote_response_schema.py`
  - Accept both v2 remote payload (PDF §7.1: `prediction_raw`,
    `optional_outputs`, `diagnostics`) and the existing flat v1
    response. Map v1 fields into v2 slots so the rest of the pipeline
    only knows v2.
- [ ] `backend/app/infrastructure/adapters/response_normalizer.py`
  - `ResponseNormalizer.normalize(...)` returns a `dict` shaped as
    `PredictionResponseV2Schema`. Populate `fallback_used`,
    `fallback_reason`, `diagnostics.warnings`,
    `diagnostics.notes=["Prototype prediction; not a field-validated
    thermoelastic simulation."]`.
  - Old flat fields (`direction_vector`, `azimuth_deg`,
    `max_displacement`, `max_temperature_perturbation`) survive only
    inside `optional_outputs.field_summary` and
    `prediction.directional_response`.
- [ ] `backend/app/domain/use_cases/predict_direction.py`
  - Before calling the model client, build the enriched v2 payload
    (PDF §4.2 / §6.1) from request + medium + `derived_quantities`.
    Backend, not the model, owns θ, distance, unit direction,
    azimuth, derived elastic constants.
  - Single `EnrichedPredictionRequest` object → reused by every client.
- [ ] `backend/app/infrastructure/clients/base.py` and
  `{pinn,fno,meshgraphnet,transformer}_client.py`
  - `build_payload(...)` adds `schema_version="2.0"`,
    `model_runtime.representation`,
    `model_runtime.requested_outputs=["temperature","displacement",
    "direction"]`.
  - No physics recomputed in the clients — they just package and ship.

**Done when:** `test_response_normalizer.py` shows both paths (remote
v2 → normalized v2, remote v1 → normalized v2) green.

---

## Phase 3 — API route + backward compatibility

- [ ] `backend/app/api/routes/predictions.py`
  - Inspect `schema_version` on the incoming request:
    - missing or `"1.0"` → run the existing v1 path (returns v1
      response, no behaviour change).
    - `"2.0"` → run the v2 path (returns v2 response).
  - Single endpoint `POST /api/v1/predictions`. No new URL.
  - Optional query flag `?contract=v2` to force v2 response even when
    request was sent as v1 (useful for frontend rollout).
- [ ] `backend/tests/test_api_smoke.py`
  - Two parametrised cases: v1 in → v1 out, v2 in → v2 out. Same
    happy-path payload.

**Done when:** old frontend builds keep working (Docker Compose smoke
test passes unchanged) AND a curl with `schema_version: "2.0"`
returns a v2 response.

---

## Phase 4 — Model services

Each service accepts v2 payload and returns the PDF §7.1 shape. Make
the change additive so a v1-shaped payload still works (status quo).

- [ ] `pinn-service/src/pinn_service/service_app.py`
  - `PINNPredictionRequest` gains v2 fields; predict returns
    `prediction_raw.temperature_k`, `displacement_m.u`,
    `displacement_m.v`. Stress/strain only into `optional_outputs`.
  - New test: `pinn-service/tests/test_api_contract_v2.py`.
- [ ] `fno-service/src/fno_service/api/routes.py`
  - Same shape. `model_runtime.representation="grid"`, `rect_2d`
    keeps `z=0` and ignores it. `field_summary` populated when
    sampling is cheap; `field_grid` only if size is safe.
  - Test: `fno-service/tests/test_api_contract_v2.py`.
- [ ] `mgn-service/src/service/api.py`
  - Real rollout when artifacts are present; otherwise deterministic
    v2 response with `diagnostics.fallback_used=true,
    fallback_reason="missing-artifact"`.
- [ ] `transformer-service/src/transformer_service/service_app.py`
  - Same v2 response. Demo/mock path explicitly tagged
    `fallback_used=true, fallback_reason="demo-mode"` if no checkpoint
    is loaded.

**Done when:** every service container starts on `docker-compose up`,
returns 200 on `/health`, and a v2 request returns a v2-shaped
response (live curl from the host).

---

## Phase 5 — Frontend

Order: request builder → renderer → cosmetic copy. Each step is shipped
and reviewed independently to keep blast radius small.

- [ ] `frontend/assets/scripts/form.js`
  - `DEMO_TEMPLATE` → v2 shape (`schema_version: "2.0"`,
    `thermal_state`, `geometry.dimension: 2`, `observation.time_s`,
    `scenario.*`). `readPayloadFromForm` reads v2.
  - Keep `applyModelDomainPolicy` but drop 3D as default for the demo.
    `rect_3d` stays available as future-extension toggle.
- [ ] `frontend/assets/scripts/validators.js`
  - `T_ref > 0 K`, `T_source >= 0 K`, `source != probe`, `time_s > 0`,
    finite coordinates. Block submit on failure with inline error.
- [ ] `frontend/assets/scripts/api.js`
  - Parse v2 response. Surface `diagnostics.warnings` and
    `model.fallback_used` to the UI layer.
- [ ] `frontend/assets/scripts/ui.js`
  - `renderResult` reads `response.prediction.thermal`,
    `response.prediction.displacement`,
    `response.geometry`, `response.model`.
  - Four cards: Thermal response / Mechanical response / Directional
    geometry / Model metadata (with fallback badge).
- [ ] `frontend/assets/scripts/charts.js`
  - SVG preview keyed off `geometry.source`/`geometry.probe`. Optional
    heatmap only when `optional_outputs.field_grid != null`.
- [ ] `frontend/assets/scripts/state.js`
  - Track `contractVersion`, `lastDerivedGeometry`, `lastDiagnostics`
    for the debug panel.
- [ ] `frontend/index.html`
  - Section labels: *Thermal input*, *Geometry*, *Observation*,
    *Simplified scenario*. Remove visible 3D fields from the default
    demo. Result section: cards listed above.

**Done when:** `docker-compose up` → open frontend → submit demo →
result panel shows T, θ, u, v, |u|, distance, azimuth, model badge,
fallback status. No JS console errors.

---

## Phase 6 — Tests, docs, demo

- [ ] `backend/tests/test_prediction_schema.py`
  - Minimal v2 request validates; `source == probe` rejected; negative
    temperatures rejected; missing `rect_2d` falls back to default.
- [ ] `backend/tests/test_response_normalizer.py`
  - Remote v2 → normalized v2. Remote v1 → normalized v2 (compat).
- [ ] `backend/tests/test_predict_direction_use_case.py`
  - Medium resolved, derived geometry attached, theta computed.
- [ ] `pinn-service/tests/test_api_contract_v2.py`,
  `fno-service/tests/test_api_contract_v2.py` — service-side checks
  per Phase 4.
- [ ] Frontend smoke (manual or Playwright if available): demo payload
  is v2, no mandatory 3D fields.
- [ ] `README.md` — add a short "API v2" section and curl example.
- [ ] `docs/api-contract-v2.md` — final, with screenshots / OpenAPI
  snippet.

---

## Acceptance criteria (PDF §12)

- [ ] **API request:** frontend sends `schema_version=2.0` with
  `material`, `thermal_state`, `geometry`, `observation`, `scenario`.
- [ ] **API response:** every model route returns
  T / θ / u / v / |u| / distance / azimuth / model metadata /
  fallback status, normalized identically across PINN, FNO, MGN,
  Transformer.
- [ ] **Physics wording:** stress and strain remain optional derived
  diagnostics; never advertised as direct predictions.
- [ ] **2D consistency:** `rect_2d` ⇒ `z=0`, `direction_z=0`, elevation
  omitted.
- [ ] **Comparison:** responses comparable across all four routes
  (units + metadata consistent).
- [ ] **Thesis safety:** no UI or doc copy claims a full
  field-validated thermoelastic simulator. The
  `diagnostics.notes[0]` line stays as the canonical disclaimer.

---

## Non-goals

- No new HTTP endpoint URL. `POST /api/v1/predictions` stays the entry
  point.
- No change to the catalog material set or numeric properties (only
  documentation/derivation fields added).
- No removal of v1 schemas in this iteration. Deprecation/removal is
  a follow-up after the frontend has been on v2 for at least one
  release.
- No introduction of 3D defaults. `rect_3d` stays gated behind a
  future-extension flag.

---

## Open questions to clear with the author before Phase 1

- Catalog enrichment: do we add `young_modulus_pa` / `poisson_ratio`
  with literature defaults per material, or compute them on the fly
  from `Vp`, `Vs`, `ρ`? The PDF allows both — pick one for the thesis
  numbers.
- Optional outputs (`field_grid`, `travel_time_s`): include from
  Phase 4 or postpone to a v2.1? PDF lists them as optional so the
  default is omit.
- Frontend rollout: serve old `frontend/` until Phase 5 ships, or
  feature-flag v2 rendering via `?contract=v2`? Recommend the second
  — smaller blast radius.
