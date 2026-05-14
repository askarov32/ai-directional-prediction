# Data Experiment Pipeline

This note captures the canonical pipeline for reproducible direct model-service experiments.

## Goal

Run the same canonical experiment cases through:

- `pinn-service`
- `mgn-service`
- `fno-service`
- `transformer-service`

Then:

- save raw requests
- save raw responses
- normalize outputs
- compute comparison metrics
- generate charts

## Why a canonical input layer is required

The public backend request contract is unified, but the direct model-service contracts and direct response formats differ.

Request differences:

- `PINN` requires `representation="physics_informed"` and `routing_hint="pinn"`
- `MGN` uses `representation="graph"` and optional `routing_hint="meshgraphnet"`
- `FNO` uses `representation="grid"`, `routing_hint="fno"`, and currently benefits from `requested_outputs` and `grid_policy`
- `Transformer` requires `representation="tokenset"` and `routing_hint="transformer"`

Response differences:

- `PINN`, `MGN`, and `Transformer` return flat postprocessed fields
- `FNO` returns nested `prediction` and `field_summary`

So the experiment pipeline should not write model-specific logic everywhere. It should use:

1. one canonical experiment-case schema
2. one mapper per model service
3. one response normalizer

## Canonical experiment-case schema

Suggested format for `artifacts/data_experiments/inputs/model_comparison_inputs.jsonl`:

```json
{
  "case_id": "case_001_sandstone",
  "material": "sandstone",
  "scenario": {
    "temperature_c": 120.0,
    "pressure_mpa": 35.0,
    "time_ms": 12.0
  },
  "source": {
    "type": "thermal_pulse",
    "x": 0.15,
    "y": 0.40,
    "z": 0.0,
    "amplitude": 1.0,
    "frequency_hz": 50.0,
    "direction": [1.0, 0.0, 0.0]
  },
  "probe": {
    "x": 0.70,
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

The generator should keep all physics inputs in this canonical form and should not store direct service payloads as the primary source.

## Mapping layer

One adapter per service:

- `canonical_to_pinn_payload(case, medium)`
- `canonical_to_mgn_payload(case, medium)`
- `canonical_to_fno_payload(case, medium)`
- `canonical_to_transformer_payload(case, medium)`

All adapters should:

- inject the resolved medium object
- set the correct `representation`
- set the correct `routing_hint`

Only FNO should currently add:

- `requested_outputs`
- `grid_policy`

## Output normalization

One function should normalize every raw response into a canonical result record:

```json
{
  "case_id": "case_001_sandstone",
  "model": "fno",
  "status": "ok",
  "service_mode": "fallback",
  "fallback_used": true,
  "material": "sandstone",
  "temperature_c": 120.0,
  "pressure_mpa": 35.0,
  "time_ms": 12.0,
  "frequency_hz": 50.0,
  "direction_x": 0.981194,
  "direction_y": 0.193022,
  "direction_z": 0.0,
  "azimuth_deg": 11.1292,
  "elevation_deg": 0.0,
  "magnitude": 1.0,
  "travel_time_ms_pred": 2.363967,
  "max_displacement": 0.00145952,
  "max_temperature_perturbation": 1.2,
  "wave_type": "fno_skeleton_fallback",
  "model_version": "fno-skeleton-fallback-v0"
}
```

If a service fails, store:

```json
{
  "case_id": "case_001_sandstone",
  "model": "pinn",
  "status": "error",
  "http_status": 503,
  "error_code": "CHECKPOINT_NOT_READY",
  "error_message": "PINN checkpoint is not loaded."
}
```

## Recommended directory layout

```text
artifacts/data_experiments/
  inputs/
    model_comparison_inputs.jsonl
  requests/
    pinn_requests.jsonl
    mgn_requests.jsonl
    fno_requests.jsonl
    transformer_requests.jsonl
  outputs/
    raw_pinn_responses.jsonl
    raw_mgn_responses.jsonl
    raw_fno_responses.jsonl
    raw_transformer_responses.jsonl
    normalized_results.jsonl
  tables/
  charts/
  reports/
```

## Recommended execution order

1. Generate canonical input cases.
2. Resolve medium presets.
3. Build direct payloads for each service.
4. Call `/health` and `/ready` once per service before the run.
5. Run all `/predict` calls.
6. Save raw responses exactly as returned.
7. Normalize all outputs into one flat table.
8. Build summary metrics and charts.
9. Save a run report with:
   - total cases
   - ok count
   - fallback count
   - error count
   - elapsed time

## Recommended first experiment grid

Start with:

- materials: `sandstone`, `basalt`
- temperatures: `20`, `120`, `220`, `300`
- pressures: `5`, `35`
- times: `6`, `12`
- frequencies: `25`, `50`

This gives a manageable first set while still showing model sensitivity.

## Important operational notes

- `FNO` currently supports only `rect_2d` and `nz=1`.
- `PINN` and `Transformer` require valid checkpoints for `ready=true`.
- `MGN` and `FNO` may return usable fallback responses if fallback is enabled.
- The experiment report should explicitly mark fallback outputs so they are not confused with checkpoint-based inference.
