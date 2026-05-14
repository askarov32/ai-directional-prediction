# FNO API Contract

## Backend To FNO Request

The backend sends an enriched payload to `POST /predict` on `fno-service`.

Current shape:

```json
{
  "medium": {
    "id": "granite",
    "name": "Granite",
    "category": "igneous",
    "properties": {
      "rho": 2650.0,
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
      "source": "mvp_catalog_extension"
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
  "representation": "grid",
  "routing_hint": "fno",
  "requested_outputs": ["direction", "field_summary"],
  "grid_policy": "service_default"
}
```

## FNO To Backend Response

Current expected shape:

```json
{
  "prediction": {
    "direction_vector": [0.821, 0.571, 0.0],
    "azimuth_deg": 34.8,
    "elevation_deg": 0.0,
    "magnitude": 0.914,
    "wave_type": "fno_checkpoint_inference",
    "travel_time_ms": 12.4
  },
  "field_summary": {
    "max_displacement": 0.001327,
    "max_temperature_perturbation": 1.742
  },
  "model_version": "fno-baseline@best_model.pth",
  "diagnostics": {
    "checkpoint_loaded": true,
    "device": "cpu",
    "input_channels": ["temperature_k"],
    "output_channels": ["temperature_k", "disp_x", "disp_y", "disp_z"]
  }
}
```

The backend normalizer consumes:

- `prediction`
- `field_summary`
- `model_version`

and ignores extra diagnostic fields.

## Error Cases

Typical service-level errors:

- `503 CHECKPOINT_NOT_READY`
- `400 UNSUPPORTED_DOMAIN`
- `500 MODEL_LOAD_FAILED`
- `500 NON_FINITE_MODEL_OUTPUT`

## Important MVP Note

The backend does not send full regular-grid tensors to the FNO service.

Instead:

1. the backend sends the enriched scenario payload;
2. `fno-service` loads its local dataset/checkpoint artifacts;
3. the service builds the model input internally.
