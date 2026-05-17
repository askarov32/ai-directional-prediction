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

## Phase 0 — Contract document (no code) — **DONE 2026-05-17**

- [x] `docs/api-contract-v2.md` — full request and response examples,
  units, field ownership table (model / backend / derived). Lifted
  from PDF §4.1, §4.2, §5, §5.1.
- [x] curl example (PDF §11) included.
- [x] "thesis-safe wording" block (PDF §13) included verbatim and
  threaded into `diagnostics.notes`.
- [x] Cross-linked with this file. Contract doc at
  [`api-contract-v2.md`](api-contract-v2.md).
- [x] Three open-question resolutions baked into the contract:
  catalog from thesis CSV, hybrid optional outputs (travel_time_s
  required, field_grid opt-in FNO-only, confidence/strain/stress
  null), `?contract=v2` frontend feature flag.

**Done when:** another contributor can implement a v2 client from
`docs/api-contract-v2.md` alone, without reading the PDF. ✅

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
  - Add `PredictionResponseV2Schema` with the nested blocks of
    PDF §5 plus a temporal-response promotion:
    `prediction.thermal`, `prediction.displacement`,
    `prediction.directional_response`,
    `prediction.temporal_response.travel_time_s` (**required**,
    moved out of optional_outputs per resolution 2026-05-17),
    `optional_outputs`, `diagnostics`, plus top-level
    `model` / `material` / `geometry`.
  - `optional_outputs` schema explicitly declares `confidence_score`,
    `field_grid`, `strain`, `stress`, `field_summary.*`.
    `confidence_score`, `strain`, `stress` are typed as nullable and
    documented to be `null` for every route in v2.
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
- [ ] `backend/data/media/catalog.json` — **replace** with a JSON
  projection of the canonical thesis table
  `AI_Termoelastic_Waves_Geology/chapters/tables/combined_geological_media_parameters.csv`
  (10 materials: granite, granodiorite, basalt, diabase, gabbro,
  sandstone, limestone, marble, schist, quartzite).
  - Carry over every field from the CSV: `rho_kg_m3`, `Vp_m_s`,
    `Vs_m_s`, `E_Pa`, `nu`, `mu_Pa`, `K_Pa`, `lambda_Pa`, `alpha_1_K`,
    `k_W_mK`, `Cp_J_kgK`, `C_J_m3K`, `gamma_Pa_K`, `porosity_percent`.
  - Each entry has `metadata.source_table:
    "combined_geological_media_parameters.csv"` for provenance.
  - Materials without `alpha_1_K` (granodiorite, basalt, diabase,
    gabbro, schist, quartzite) are flagged
    `thermoelastic_supported: false`. The backend rejects
    `POST /api/v1/predictions` for these media with HTTP 400 and a
    clear reason; the frontend disables the submit button when one
    is selected for a thermoelastic scenario.
  - `derived_quantities.derive_elastic_constants(...)` does **not**
    recompute `E`, `nu`, `mu`, `K`, `lambda` — it reads them from the
    catalog. On startup it verifies `|E_csv - 9Kμ/(3K+μ)| / E_csv <
    0.01` and logs a warning on mismatch (CSV-typo guard).

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
  - Always populate `prediction.temporal_response.travel_time_s` from
    whatever the remote returned (`travel_time_ms` / 1000 if v1 flat,
    `prediction_raw.travel_time_s` if v2). Fail loudly if missing —
    travel-time is required.
  - Old flat fields (`direction_vector`, `azimuth_deg`,
    `max_displacement`, `max_temperature_perturbation`) survive only
    inside `optional_outputs.field_summary` and
    `prediction.directional_response`.
  - `optional_outputs.field_grid` is included only when the request
    asked for it via `model_runtime.requested_outputs` AND the
    remote returned it; otherwise `null`. `confidence_score`,
    `strain`, `stress` are always `null` in v2.
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
    `displacement_m.v`, **and `prediction_raw.travel_time_s`**
    (required for all services in v2). Stress/strain stay `null`
    in `optional_outputs`.
  - New test: `pinn-service/tests/test_api_contract_v2.py`.
- [ ] `fno-service/src/fno_service/api/routes.py`
  - Same shape, including `prediction_raw.travel_time_s`.
    `model_runtime.representation="grid"`, `rect_2d` keeps `z=0`
    and ignores it. `field_summary` populated when sampling is
    cheap. **`field_grid` is the only route that may emit it**,
    and only when the caller listed `"field_grid"` in
    `model_runtime.requested_outputs` AND grid size ≤ 128×128.
  - Test: `fno-service/tests/test_api_contract_v2.py`.
- [ ] `mgn-service/src/service/api.py`
  - Real rollout when artifacts are present; otherwise deterministic
    v2 response with `diagnostics.fallback_used=true,
    fallback_reason="missing-artifact"`.
  - Either path must populate `prediction_raw.travel_time_s`
    (fallback uses analytical `r/V_p`). `field_grid` always `null`.
- [ ] `transformer-service/src/transformer_service/service_app.py`
  - Same v2 response, including required
    `prediction_raw.travel_time_s`. Demo/mock path explicitly
    tagged `fallback_used=true, fallback_reason="demo-mode"` if
    no checkpoint is loaded. `field_grid` always `null`.

**Done when:** every service container starts on `docker-compose up`,
returns 200 on `/health`, and a v2 request returns a v2-shaped
response (live curl from the host).

---

## Phase 5 — Frontend

Order: feature flag → request builder → renderer → cosmetic copy.
Each step is shipped and reviewed independently to keep blast radius
small. Default stays v1 until defense; flag flips after one week green.

- [ ] `frontend/assets/scripts/state.js` **(first — gate every other
  change behind it)**
  - Export `CONTRACT_VERSION` derived from the URL:
    `new URLSearchParams(location.search).get("contract") === "v2"
    ? "2.0" : "1.0"`.
  - Track `contractVersion`, `lastDerivedGeometry`, `lastDiagnostics`
    for the debug panel.
  - Every v2-specific branch in `form.js` / `ui.js` / `charts.js` /
    `api.js` is gated by `CONTRACT_VERSION === "2.0"`. The v1 path
    stays byte-identical to today's behaviour.
- [ ] `frontend/index.html`
  - Add a small "Try v2 contract" link in the header. Hard-coded
    `href="?contract=v2"`; when already on v2 it becomes
    "Back to v1 contract" with `href="?"`.
  - No other changes in this step.
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
- [ ] `frontend/index.html` — v2-only section labels
  - Inside the v2 branch only: section labels become *Thermal
    input*, *Geometry*, *Observation*, *Simplified scenario*.
    Remove visible 3D fields. Result section: cards listed above.
  - The v1 markup stays present and untouched so `?contract=v1`
    (or no flag) keeps the current demo working.

### Default-flip rollout (after Phase 5 + tests are green)

- [ ] Wait one week after Phase 6 ships with no production errors
  attributable to v2.
- [ ] Flip the default in `state.js`: v2 becomes the default,
  `?contract=v1` keeps the old path alive for one more release.
- [ ] One release later: delete the v1 branches (forms, renderer,
  styles), delete v1 schemas in `backend/app/schemas/prediction.py`,
  and delete the `?contract=` query parameter handling.

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

- ~~Catalog enrichment~~ **Resolved 2026-05-17.** Catalog is replaced
  by a JSON projection of
  `chapters/tables/combined_geological_media_parameters.csv`. All
  elastic moduli come straight from the CSV (already derived from
  midpoint Vₚ/Vₛ/ρ under isotropic assumptions). Materials with
  missing `alpha_1_K` are marked `thermoelastic_supported: false`
  and rejected with HTTP 400 for thermoelastic requests. No
  literature defaults are injected.
- ~~Optional outputs (`field_grid`, `travel_time_s`)~~ **Resolved
  2026-05-17. Hybrid (option C):**
  - `travel_time_s` is **promoted out of optional_outputs** into a
    required temporal output at
    `prediction.temporal_response.travel_time_s`. All four model
    routes must populate it (every service already computes it
    internally). Rationale: travel-time accuracy vs the analytical
    reference `t = r/V_p` is the headline accuracy chart in the
    thesis defence — it must not live behind an `optional` gate.
  - `optional_outputs.field_summary.{max_displacement_m,
    max_temperature_perturbation_k}` stays for backward
    compatibility with v1 consumers.
  - `optional_outputs.field_grid` is declared in the schema but
    **opt-in only**: the client must list `"field_grid"` inside
    `model_runtime.requested_outputs` to receive it. Only the FNO
    route returns it in v2; other routes leave it `null`. Capped
    at Nx×Ny ≤ 128×128 so a single response stays under ~1.5 MB.
  - `optional_outputs.confidence_score` is declared but always
    `null` in v2 — implementation deferred to v2.1.
  - `optional_outputs.strain` and `optional_outputs.stress` are
    declared but always `null` in v2 (PDF §3 explicitly forbids
    making them mandatory predictions in this prototype).
- ~~Frontend rollout~~ **Resolved 2026-05-17. Feature flag (option B):**
  one frontend bundle, behaviour gated by a `?contract=v2` query
  parameter. Default stays v1 until the thesis defense is done plus
  one week of green runs, then the default flips to v2 and v1 lives
  behind `?contract=v1` for one more release before removal. No
  parallel `/v2/` route, no duplicated `index.html`. Backend Phase 3
  already accepts both contracts, so the flag is purely cosmetic on
  the wire.
