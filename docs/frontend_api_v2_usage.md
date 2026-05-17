# Frontend API v2 Usage

This document describes the current frontend behavior after the API v2 UI redesign.

## Scope

The frontend is a research prototype for AI-assisted directional prediction and model comparison.
It is not a field-validated thermoelastic simulation.

## What The Frontend Sends

The current UI sends API Contract v2 payloads only.

```json
{
  "schema_version": "2.0",
  "model": "pinn",
  "medium_id": "sandstone",
  "geometry": {
    "dimension": 2,
    "source": { "x_m": 0.15, "y_m": 0.40 },
    "probe": { "x_m": 0.70, "y_m": 0.55 }
  },
  "observation": {
    "time_s": 0.012
  },
  "scenario": {
    "thermal_source_type": "point",
    "mechanical_constraint": "free",
    "boundary_condition_type": "prototype_simplified"
  }
}
```

The UI does not send:

- `z` coordinates
- `dimension: 3`
- custom domain size or resolution
- `frequency_hz`
- source amplitude or explicit direction vectors
- old temperature/pressure form fields
- backend-owned thermal overrides such as `reference_temperature_k`

## Geometry Constraints

The visible UI is fixed to a planar `1 m x 1 m` domain.

- `0 <= source.x_m <= 1`
- `0 <= source.y_m <= 1`
- `0 <= probe.x_m <= 1`
- `0 <= probe.y_m <= 1`
- source and probe cannot be identical
- `observation.time_s > 0`

Validation is shown inline in the form.

## Medium Catalog Compatibility

The backend `/media` route may still expose the legacy id `sandstone_medium`.
The frontend normalizes that value to the API v2 id `sandstone` before sending predictions, so the visible sandstone option remains usable in the current UI.

## What The Frontend Renders

The result workspace shows:

- interactive 2D source-to-probe SVG geometry
- thermal response card
- displacement response card
- directional response card
- temporal response card
- model metadata card
- diagnostics and fallback panel
- collapsible debug JSON

## Optional Heatmap

The spatial heatmap is optional.

- If `response.optional_outputs.field_grid` exists, the frontend renders a heatmap with channel switching, min/max statistics, and hover values.
- If `field_grid` is `null` or absent, the frontend shows a neutral message instead of a broken chart.

This is expected for PINN, MeshGraphNet, and Transformer in the current prototype flow.

## Current UI Limitations

- The visible UI is 2D-only.
- The backend `/models` route may still describe model services with older capability notes, but the frontend does not expose those 3D hints in the form.
- The heatmap depends on optional route output and is not guaranteed for every model request.
- The frontend emphasizes normalized prediction response and model comparison, not raw service payloads as the main view.
